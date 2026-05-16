import os, secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from contextlib import asynccontextmanager

from google.oauth2 import id_token
from google.auth.transport import requests
from pydantic import BaseModel

# FastAPI recibe un JSON con el token
class GoogleTokenRequest(BaseModel):
    id_token: str

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import Column, Integer, String, Boolean, DateTime, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
import bcrypt
from jose import jwt, JWTError
import re

# ─── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./auth.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
APP_NAME = os.getenv("APP_NAME", "FastAPI Auth")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

bearer = HTTPBearer(auto_error=False)

# ─── Database ─────────────────────────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200), default="")
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class RevokedToken(Base):
    __tablename__ = "revoked_tokens"
    id = Column(Integer, primary_key=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)
    revoked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# ─── Schemas ──────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str = ""

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,30}$", v):
            raise ValueError("Username: 3-30 chars, letters/digits/underscores only")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

class LoginRequest(BaseModel):
    login: str  # email or username
    password: str

class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    is_active: bool
    is_admin: bool
    created_at: datetime

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    payload["iat"] = datetime.now(timezone.utc)
    payload["jti"] = secrets.token_urlsafe(32)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_tokens(user_id: int) -> TokenResponse:
    access = create_token(
        {"sub": str(user_id), "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE),
    )
    refresh = create_token(
        {"sub": str(user_id), "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE),
    )
    return TokenResponse(access_token=access, refresh_token=refresh, expires_in=ACCESS_TOKEN_EXPIRE * 60)

async def get_db():
    async with SessionLocal() as session:
        yield session

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        jti = payload.get("jti")
        revoked = await db.execute(select(RevokedToken).where(RevokedToken.jti == jti))
        if revoked.scalar_one_or_none():
            raise HTTPException(status_code=401, detail="Token revoked")
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user

async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ─── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(title=APP_NAME, lifespan=lifespan)

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.post("/api/auth/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(User).where((User.email == req.email) | (User.username == req.username))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email or username already taken")
    user = User(
        email=req.email,
        username=req.username,
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return create_tokens(user.id)

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where((User.email == req.login) | (User.username == req.login))
    )
    user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")
    return create_tokens(user.id)


GOOGLE_CLIENT_ID = "437346837668-4aed9bubi45tsd62an8tsvfum47jgvao.apps.googleusercontent.com"

@app.post("/api/auth/google", response_model=TokenResponse)
async def google_auth(req: GoogleTokenRequest, db: AsyncSession = Depends(get_db)):
    try:
        id_info = id_token.verify_oauth2_token(
            req.id_token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        email = id_info.get("email")
        full_name = id_info.get("name", "")
        
        username = email.split("@")[0]

        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                email=email,
                username=username,
                hashed_password=hash_password("google_oauth_secure_password_placeholder"),
                full_name=full_name,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

        return create_tokens(user.id)

    except ValueError:
        raise HTTPException(status_code=401, detail="Token de Google inválido o expirado")

@app.post("/api/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json()
    token = body.get("refresh_token", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        jti = payload.get("jti")
        revoked = await db.execute(select(RevokedToken).where(RevokedToken.jti == jti))
        if revoked.scalar_one_or_none():
            raise HTTPException(status_code=401, detail="Token revoked")
        # Revoke old refresh token
        db.add(RevokedToken(jti=jti))
        await db.commit()
        return create_tokens(int(payload["sub"]))
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.post("/api/auth/logout")
async def logout(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        db.add(RevokedToken(jti=payload["jti"]))
        await db.commit()
    except JWTError:
        pass
    return {"message": "Logged out"}

# ─── User Routes ──────────────────────────────────────────────────────────────
@app.get("/api/users/me", response_model=UserResponse)
async def get_profile(user: User = Depends(get_current_user)):
    return user

@app.patch("/api/users/me", response_model=UserResponse)
async def update_profile(
    req: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if req.username and req.username != user.username:
        if not re.match(r"^[a-zA-Z0-9_]{3,30}$", req.username):
            raise HTTPException(status_code=422, detail="Invalid username format")
        exists = await db.execute(select(User).where(User.username == req.username))
        if exists.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username taken")
        user.username = req.username
    if req.full_name is not None:
        user.full_name = req.full_name
    user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return user

@app.post("/api/users/me/change-password")
async def change_password(
    req: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(req.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")
    user.hashed_password = hash_password(req.new_password)
    user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return {"message": "Password changed"}

@app.delete("/api/users/me")
async def delete_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await db.execute(delete(User).where(User.id == user.id))
    await db.commit()
    return {"message": "Account deleted"}

# ─── Admin Routes ─────────────────────────────────────────────────────────────
@app.get("/api/admin/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0, limit: int = 50,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

@app.patch("/api/admin/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    target.is_active = not target.is_active
    await db.commit()
    return {"id": target.id, "is_active": target.is_active}

@app.patch("/api/admin/users/{user_id}/toggle-admin")
async def toggle_user_admin(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change own admin status")
    target.is_admin = not target.is_admin
    await db.commit()
    return {"id": target.id, "is_admin": target.is_admin}

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "app": APP_NAME}

# ─── Frontend ─────────────────────────────────────────────────────────────────
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + APP_NAME + """ - Auth Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0f172a;--surface:#1e293b;--surface2:#334155;--primary:#6366f1;--primary-h:#818cf8;--green:#22c55e;--red:#ef4444;--text:#f1f5f9;--muted:#94a3b8;--radius:12px}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center}
.container{width:100%;max-width:460px;padding:20px}
.card{background:var(--surface);border-radius:var(--radius);padding:32px;box-shadow:0 25px 50px -12px rgba(0,0,0,.4)}
h1{font-size:1.5rem;text-align:center;margin-bottom:4px}
.subtitle{text-align:center;color:var(--muted);font-size:.875rem;margin-bottom:24px}
.tabs{display:flex;gap:4px;margin-bottom:24px;background:var(--bg);border-radius:8px;padding:4px}
.tab{flex:1;padding:10px;text-align:center;border:none;background:none;color:var(--muted);cursor:pointer;border-radius:6px;font-size:.875rem;font-weight:600;transition:.2s}
.tab.active{background:var(--primary);color:#fff}
.form-group{margin-bottom:16px}
.form-group label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:6px;font-weight:500}
.form-group input{width:100%;padding:10px 14px;background:var(--bg);border:1px solid var(--surface2);border-radius:8px;color:var(--text);font-size:.9rem;outline:none;transition:.2s}
.form-group input:focus{border-color:var(--primary)}
.btn{width:100%;padding:12px;background:var(--primary);color:#fff;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;transition:.2s;margin-top:8px}
.btn:hover{background:var(--primary-h)}
.btn-danger{background:var(--red)}
.btn-danger:hover{background:#dc2626}
.btn-sm{width:auto;padding:8px 16px;font-size:.8rem;margin-top:0}
.alert{padding:12px 16px;border-radius:8px;font-size:.85rem;margin-bottom:16px;display:none}
.alert.error{background:rgba(239,68,68,.15);color:#fca5a5;display:block}
.alert.success{background:rgba(34,197,94,.15);color:#86efac;display:block}
.profile-section{text-align:center}
.avatar{width:72px;height:72px;background:var(--primary);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;margin:0 auto 16px;color:#fff}
.profile-info{margin:16px 0}
.profile-info .field{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--surface2);font-size:.875rem}
.profile-info .field .label{color:var(--muted)}
.profile-actions{display:flex;flex-direction:column;gap:8px;margin-top:20px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:600;margin-left:8px}
.badge-admin{background:rgba(99,102,241,.2);color:var(--primary-h)}
.badge-active{background:rgba(34,197,94,.2);color:var(--green)}
.hidden{display:none!important}
.admin-panel{margin-top:16px;padding-top:16px;border-top:1px solid var(--surface2)}
.user-row{display:flex;justify-content:space-between;align-items:center;padding:10px;background:var(--bg);border-radius:8px;margin-bottom:8px;font-size:.85rem}
.user-row .actions{display:flex;gap:4px}
.btn-icon{background:var(--surface2);border:none;color:var(--text);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:.75rem;transition:.2s}
.btn-icon:hover{background:var(--primary)}
.api-ref{margin-top:16px;font-size:.75rem;color:var(--muted);text-align:center}
.api-ref a{color:var(--primary-h);text-decoration:none}
.edit-form{margin-top:16px}
</style>
</head>
<body>
<div class="container">
<div class="card">
<h1>🔐 """ + APP_NAME + """</h1>
<p class="subtitle">Authentication & User Management</p>
<div id="alert" class="alert"></div>

<!-- Auth Section -->
<div id="auth-section">
<div class="tabs">
  <button class="tab active" onclick="switchTab('login')">Sign In</button>
  <button class="tab" onclick="switchTab('register')">Sign Up</button>
</div>
<form id="login-form" onsubmit="return handleLogin(event)">
  <div class="form-group"><label>Email or Username</label><input id="login-id" required placeholder="you@example.com"></div>
  <div class="form-group"><label>Password</label><input id="login-pw" type="password" required placeholder="••••••••"></div>
  <button class="btn" type="submit">Sign In</button>
</form>
<form id="register-form" class="hidden" onsubmit="return handleRegister(event)">
  <div class="form-group"><label>Full Name</label><input id="reg-name" placeholder="Jane Doe"></div>
  <div class="form-group"><label>Username</label><input id="reg-user" required placeholder="janedoe"></div>
  <div class="form-group"><label>Email</label><input id="reg-email" type="email" required placeholder="you@example.com"></div>
  <div class="form-group"><label>Password (min 8 chars)</label><input id="reg-pw" type="password" required minlength="8" placeholder="••••••••"></div>
  <button class="btn" type="submit">Create Account</button>
</form>
<div class="api-ref">API Docs: <a href="/docs" target="_blank">/docs</a> · <a href="/redoc" target="_blank">/redoc</a></div>
</div>

<!-- Dashboard Section -->
<div id="dash-section" class="hidden">
<div class="profile-section" id="profile-view">
  <div class="avatar" id="avatar">?</div>
  <h2 id="dash-name">User</h2>
  <p style="color:var(--muted);font-size:.85rem" id="dash-email">email</p>
  <div id="badges"></div>
  <div class="profile-info" id="profile-info"></div>
  <div class="profile-actions">
    <button class="btn" onclick="showEditProfile()">Edit Profile</button>
    <button class="btn" style="background:var(--surface2)" onclick="showChangePassword()">Change Password</button>
    <button class="btn btn-danger" onclick="handleLogout()">Sign Out</button>
    <button class="btn btn-danger" style="font-size:.8rem;background:transparent;color:var(--red);border:1px solid var(--red)" onclick="handleDeleteAccount()">Delete Account</button>
  </div>
</div>

<div id="edit-profile" class="hidden edit-form">
  <h3 style="margin-bottom:12px">Edit Profile</h3>
  <div class="form-group"><label>Full Name</label><input id="edit-name"></div>
  <div class="form-group"><label>Username</label><input id="edit-username"></div>
  <div style="display:flex;gap:8px">
    <button class="btn" onclick="handleUpdateProfile()">Save</button>
    <button class="btn" style="background:var(--surface2)" onclick="hideEditForms()">Cancel</button>
  </div>
</div>

<div id="change-pw" class="hidden edit-form">
  <h3 style="margin-bottom:12px">Change Password</h3>
  <div class="form-group"><label>Current Password</label><input id="cur-pw" type="password"></div>
  <div class="form-group"><label>New Password</label><input id="new-pw" type="password" minlength="8"></div>
  <div style="display:flex;gap:8px">
    <button class="btn" onclick="handleChangePassword()">Update</button>
    <button class="btn" style="background:var(--surface2)" onclick="hideEditForms()">Cancel</button>
  </div>
</div>

<div id="admin-panel" class="hidden admin-panel">
  <h3 style="margin-bottom:12px">👑 Admin Panel</h3>
  <div id="users-list"></div>
</div>
</div>

</div>
</div>
<script>
let tokens = JSON.parse(localStorage.getItem('auth_tokens') || 'null');
let currentUser = null;

const $ = id => document.getElementById(id);
const api = async (path, opts = {}) => {
  const headers = {'Content-Type': 'application/json', ...opts.headers};
  if (tokens?.access_token) headers['Authorization'] = `Bearer ${tokens.access_token}`;
  const res = await fetch(path, {...opts, headers});
  if (res.status === 401 && tokens?.refresh_token && !opts._retry) {
    const ref = await fetch('/api/auth/refresh', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({refresh_token: tokens.refresh_token})});
    if (ref.ok) { tokens = await ref.json(); localStorage.setItem('auth_tokens', JSON.stringify(tokens)); return api(path, {...opts, _retry: true}); }
    else { logout(); throw new Error('Session expired'); }
  }
  const data = res.status !== 204 ? await res.json() : {};
  if (!res.ok) throw new Error(data.detail || 'Request failed');
  return data;
};

function alert(msg, type='error') { const el = $('alert'); el.className = `alert ${type}`; el.textContent = msg; setTimeout(() => el.className = 'alert', 4000); }
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i) => { t.classList.toggle('active', (tab==='login'?0:1)===i); });
  $('login-form').classList.toggle('hidden', tab!=='login');
  $('register-form').classList.toggle('hidden', tab!=='register');
}

async function handleLogin(e) {
  e.preventDefault();
  try { tokens = await api('/api/auth/login', {method:'POST', body: JSON.stringify({login:$('login-id').value, password:$('login-pw').value})}); localStorage.setItem('auth_tokens', JSON.stringify(tokens)); await loadDashboard(); } catch(err) { alert(err.message); }
  return false;
}
async function handleRegister(e) {
  e.preventDefault();
  try { tokens = await api('/api/auth/register', {method:'POST', body: JSON.stringify({email:$('reg-email').value, username:$('reg-user').value, password:$('reg-pw').value, full_name:$('reg-name').value})}); localStorage.setItem('auth_tokens', JSON.stringify(tokens)); await loadDashboard(); } catch(err) { alert(err.message); }
  return false;
}
async function loadDashboard() {
  try {
    currentUser = await api('/api/users/me');
    $('auth-section').classList.add('hidden');
    $('dash-section').classList.remove('hidden');
    $('avatar').textContent = (currentUser.full_name || currentUser.username)[0].toUpperCase();
    $('dash-name').textContent = currentUser.full_name || currentUser.username;
    $('dash-email').textContent = currentUser.email;
    $('badges').innerHTML = (currentUser.is_admin ? '<span class="badge badge-admin">ADMIN</span>' : '') + '<span class="badge badge-active">ACTIVE</span>';
    $('profile-info').innerHTML = `
      <div class="field"><span class="label">Username</span><span>@${currentUser.username}</span></div>
      <div class="field"><span class="label">Member since</span><span>${new Date(currentUser.created_at).toLocaleDateString()}</span></div>
      <div class="field"><span class="label">User ID</span><span>#${currentUser.id}</span></div>`;
    hideEditForms();
    if (currentUser.is_admin) { $('admin-panel').classList.remove('hidden'); loadAdminUsers(); }
    else { $('admin-panel').classList.add('hidden'); }
  } catch(err) { logout(); }
}
function showEditProfile() { hideEditForms(); $('edit-profile').classList.remove('hidden'); $('edit-name').value = currentUser.full_name; $('edit-username').value = currentUser.username; $('profile-view').querySelector('.profile-actions').classList.add('hidden'); }
function showChangePassword() { hideEditForms(); $('change-pw').classList.remove('hidden'); $('profile-view').querySelector('.profile-actions').classList.add('hidden'); }
function hideEditForms() { $('edit-profile').classList.add('hidden'); $('change-pw').classList.add('hidden'); $('profile-view').querySelector('.profile-actions').classList.remove('hidden'); }

async function handleUpdateProfile() {
  try { await api('/api/users/me', {method:'PATCH', body: JSON.stringify({full_name:$('edit-name').value, username:$('edit-username').value})}); alert('Profile updated!', 'success'); await loadDashboard(); } catch(err) { alert(err.message); }
}
async function handleChangePassword() {
  try { await api('/api/users/me/change-password', {method:'POST', body: JSON.stringify({current_password:$('cur-pw').value, new_password:$('new-pw').value})}); alert('Password changed!', 'success'); hideEditForms(); $('cur-pw').value=''; $('new-pw').value=''; } catch(err) { alert(err.message); }
}
async function handleLogout() {
  try { await api('/api/auth/logout', {method:'POST'}); } catch(e) {}
  logout();
}
function logout() { tokens = null; currentUser = null; localStorage.removeItem('auth_tokens'); $('auth-section').classList.remove('hidden'); $('dash-section').classList.add('hidden'); }
async function handleDeleteAccount() {
  if (!confirm('Permanently delete your account? This cannot be undone.')) return;
  try { await api('/api/users/me', {method:'DELETE'}); alert('Account deleted', 'success'); logout(); } catch(err) { alert(err.message); }
}
async function loadAdminUsers() {
  try {
    const users = await api('/api/admin/users');
    $('users-list').innerHTML = users.map(u => `
      <div class="user-row">
        <span>@${u.username} ${u.is_admin?'👑':''} ${!u.is_active?'🚫':''}</span>
        <div class="actions">
          <button class="btn-icon" onclick="toggleActive(${u.id})" title="Toggle active">${u.is_active?'🔒':'🔓'}</button>
          <button class="btn-icon" onclick="toggleAdmin(${u.id})" title="Toggle admin">👑</button>
        </div>
      </div>`).join('');
  } catch(err) { console.error(err); }
}
async function toggleActive(id) { try { await api(`/api/admin/users/${id}/toggle-active`, {method:'PATCH'}); loadAdminUsers(); } catch(err) { alert(err.message); } }
async function toggleAdmin(id) { try { await api(`/api/admin/users/${id}/toggle-admin`, {method:'PATCH'}); loadAdminUsers(); } catch(err) { alert(err.message); } }

// Auto-login if tokens exist
if (tokens) loadDashboard();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def frontend():
    return FRONTEND_HTML

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
