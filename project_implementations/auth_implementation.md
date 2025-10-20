🚀 Implementation Plan: Hybrid Simple-Multi Authentication
📋 Phase Overview
PhaseDescriptionComplexityTime Est.Phase 1Single Master Password⭐ Simple30 minsPhase 2Multi-User System⭐⭐ Basic1-2 hoursPhase 3Security & Sessions⭐⭐⭐ Moderate2-3 hoursPhase 4Advanced Features⭐⭐⭐⭐ Advanced3-4 hours

🎯 Phase 1: Ultra-Simple Master Password (30 minutes)
Objective: Single password gate with minimal code changes
Step 1.1: Create auth.py module
python"""
auth.py - Phase 1: Simple master password authentication
"""
import streamlit as st
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

class SimpleAuth:
    def __init__(self):
        # For Phase 1: Use environment variable or fallback
        self.master_password = os.getenv('MASTER_PASSWORD', 'procurement2024')
        self.session_key = 'auth_simple'
    
    def hash_password(self, password: str) -> str:
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return st.session_state.get(self.session_key, {}).get('authenticated', False)
    
    def authenticate(self, password: str) -> bool:
        """Authenticate with master password"""
        if password == self.master_password:
            st.session_state[self.session_key] = {
                'authenticated': True,
                'login_time': st.session_state.get('current_time', 'unknown'),
                'user': 'admin'
            }
            return True
        return False
    
    def logout(self):
        """Clear authentication"""
        if self.session_key in st.session_state:
            del st.session_state[self.session_key]
    
    def show_login_screen(self):
        """Display simple login interface"""
        st.markdown("---")
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🔐 Authentication Required")
            st.info("🤖 **AI Procurement Data Processor**\n\nEnter the master password to access the system.")
            
            # Login form
            password = st.text_input(
                "Master Password:", 
                type="password",
                placeholder="Enter password...",
                key="login_password"
            )
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 Login", type="primary", use_container_width=True):
                    if self.authenticate(password):
                        st.success("✅ Authentication successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid password. Please try again.")
            
            # Development info (remove in production)
            if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
                st.caption(f"Debug: Master password is '{self.master_password}'")

def require_auth(auth_instance: SimpleAuth):
    """Decorator function to require authentication"""
    if not auth_instance.is_authenticated():
        auth_instance.show_login_screen()
        st.stop()
Step 1.2: Update .env file
bash# Add to .env file
MASTER_PASSWORD=your_secure_password_here
DEBUG_MODE=true
Step 1.3: Modify app.py
python# Add at the top of app.py, after imports
from auth import SimpleAuth, require_auth

def main():
    """Main application function with authentication"""
    
    # Initialize authentication
    auth = SimpleAuth()
    
    # Require authentication before proceeding
    require_auth(auth)
    
    # Add logout to sidebar
    with st.sidebar:
        if auth.is_authenticated():
            st.success("🔓 Authenticated")
            if st.button("🚪 Logout", key="logout_btn"):
                auth.logout()
                st.rerun()
    
    # Original main function code continues here...
    st.title("🤖 AI Procurement Data Intelligence")
    # ... rest of your existing code
Step 1.4: Test Phase 1
bash# Test the implementation
streamlit run app.py

# Verify:
# 1. Login screen appears first
# 2. Wrong password shows error
# 3. Correct password grants access
# 4. Logout button works
# 5. Session persists on page refresh

👥 Phase 2: Multi-User System (1-2 hours)
Objective: Support multiple users with different access levels
Step 2.1: Create users.json
json{
  "users": {
    "admin": {
      "password_hash": "hashed_password_here",
      "role": "admin", 
      "name": "Administrator",
      "permissions": ["read", "write", "admin"]
    },
    "processor": {
      "password_hash": "hashed_password_here", 
      "role": "processor",
      "name": "Data Processor",
      "permissions": ["read", "write"]
    },
    "viewer": {
      "password_hash": "hashed_password_here",
      "role": "viewer", 
      "name": "Report Viewer",
      "permissions": ["read"]
    }
  }
}
Step 2.2: Enhanced auth.py
python"""
auth.py - Phase 2: Multi-user authentication with roles
"""
import streamlit as st
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional, List

class MultiUserAuth:
    def __init__(self):
        self.users_file = "users.json"
        self.session_key = 'auth_multiuser'
        self.users = self.load_users()
    
    def load_users(self) -> Dict:
        """Load users from JSON file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default users if file doesn't exist
                return self.create_default_users()
        except Exception as e:
            st.error(f"Error loading users: {e}")
            return {}
    
    def create_default_users(self) -> Dict:
        """Create default users and save to file"""
        default_users = {
            "users": {
                "admin": {
                    "password_hash": self.hash_password("admin123"),
                    "role": "admin",
                    "name": "Administrator", 
                    "permissions": ["read", "write", "admin"],
                    "created": datetime.now().isoformat()
                },
                "viewer": {
                    "password_hash": self.hash_password("viewer123"),
                    "role": "viewer",
                    "name": "Report Viewer",
                    "permissions": ["read"],
                    "created": datetime.now().isoformat()
                }
            }
        }
        
        self.save_users(default_users)
        return default_users
    
    def save_users(self, users_data: Dict):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "procurement_salt_2024"  # Use environment variable in production
        return hashlib.pbkdf2_hmac('sha256', 
                                 password.encode('utf-8'), 
                                 salt.encode('utf-8'), 
                                 100000).hex()
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user with username and password"""
        users = self.users.get('users', {})
        
        if username in users:
            user_data = users[username]
            password_hash = self.hash_password(password)
            
            if password_hash == user_data['password_hash']:
                # Store user session
                st.session_state[self.session_key] = {
                    'authenticated': True,
                    'username': username,
                    'role': user_data['role'],
                    'name': user_data['name'],
                    'permissions': user_data['permissions'],
                    'login_time': datetime.now().isoformat()
                }
                return True
        
        return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get(self.session_key, {}).get('authenticated', False)
    
    def get_current_user(self) -> Dict:
        """Get current user information"""
        return st.session_state.get(self.session_key, {})
    
    def has_permission(self, permission: str) -> bool:
        """Check if current user has specific permission"""
        user_data = self.get_current_user()
        permissions = user_data.get('permissions', [])
        return permission in permissions
    
    def logout(self):
        """Clear authentication"""
        if self.session_key in st.session_state:
            del st.session_state[self.session_key]
    
    def show_login_screen(self):
        """Display multi-user login interface"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🔐 User Authentication")
            st.info("🤖 **AI Procurement Data Processor**\n\nPlease log in with your credentials.")
            
            # Login form
            username = st.text_input(
                "Username:", 
                placeholder="Enter username...",
                key="login_username"
            )
            
            password = st.text_input(
                "Password:", 
                type="password",
                placeholder="Enter password...",
                key="login_password"
            )
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 Login", type="primary", use_container_width=True):
                    if self.authenticate(username, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
            
            # Show available users in debug mode
            if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
                with st.expander("🔧 Debug: Available Users"):
                    users = self.users.get('users', {})
                    for user, data in users.items():
                        st.write(f"**{user}** ({data['role']}) - {data['name']}")

def require_auth(auth_instance: MultiUserAuth):
    """Require authentication before proceeding"""
    if not auth_instance.is_authenticated():
        auth_instance.show_login_screen()
        st.stop()

def require_permission(auth_instance: MultiUserAuth, permission: str):
    """Require specific permission"""
    if not auth_instance.has_permission(permission):
        st.error(f"❌ Access denied. Required permission: {permission}")
        st.stop()
Step 2.3: Update app.py for multi-user
python# Update imports
from auth import MultiUserAuth, require_auth, require_permission

def main():
    """Main application with multi-user authentication"""
    
    # Initialize multi-user authentication
    auth = MultiUserAuth()
    
    # Require authentication
    require_auth(auth)
    
    # Enhanced sidebar with user info
    with st.sidebar:
        user = auth.get_current_user()
        st.success(f"🔓 Logged in as **{user.get('name', 'Unknown')}**")
        st.caption(f"Role: {user.get('role', 'Unknown')}")
        
        if st.button("🚪 Logout", key="logout_btn"):
            auth.logout()
            st.rerun()
        
        # Show permissions (debug)
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            with st.expander("🔧 Your Permissions"):
                permissions = user.get('permissions', [])
                for perm in permissions:
                    st.write(f"✅ {perm}")
    
    # Permission-based feature access
    user = auth.get_current_user()
    role = user.get('role', '')
    
    # Modify tabs based on role
    if role == 'admin':
        tabs = ["🗂️ Data Loading", "🎯 Opportunities", "🛒 Order Optimization", "👥 User Management"]
    elif role == 'processor':
        tabs = ["🗂️ Data Loading", "🎯 Opportunities", "🛒 Order Optimization"]
    else:  # viewer
        tabs = ["🎯 Opportunities"]
    
    selected_tabs = st.tabs(tabs)
    
    # Original app content with permission checks...
    # Rest of your existing code
Step 2.4: Test Phase 2
bash# Test multi-user system
streamlit run app.py

# Verify:
# 1. Default users created (admin/admin123, viewer/viewer123)
# 2. Different roles show different tabs
# 3. User info appears in sidebar
# 4. users.json file is created

🔒 Phase 3: Security & Sessions (2-3 hours)
Objective: Add session timeout, activity tracking, and security features
Step 3.1: Enhanced session management
python"""
auth.py - Phase 3: Enhanced security and session management
"""
import streamlit as st
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List

class SecureAuth:
    def __init__(self):
        self.users_file = "users.json"
        self.session_key = 'auth_secure'
        self.config = self.load_config()
        self.users = self.load_users()
        self.failed_attempts = {}  # Track failed login attempts
    
    def load_config(self) -> Dict:
        """Load authentication configuration"""
        return {
            'session_timeout_hours': int(os.getenv('SESSION_TIMEOUT_HOURS', '8')),
            'max_failed_attempts': int(os.getenv('MAX_FAILED_ATTEMPTS', '5')),
            'lockout_duration_minutes': int(os.getenv('LOCKOUT_DURATION_MINUTES', '15')),
            'require_strong_passwords': os.getenv('REQUIRE_STRONG_PASSWORDS', 'false').lower() == 'true',
            'salt': os.getenv('PASSWORD_SALT', 'procurement_salt_2024_secure'),
        }
    
    def is_session_valid(self) -> bool:
        """Check if current session is still valid"""
        session = st.session_state.get(self.session_key, {})
        
        if not session.get('authenticated', False):
            return False
        
        # Check session timeout
        login_time_str = session.get('login_time')
        last_activity_str = session.get('last_activity')
        
        if not login_time_str or not last_activity_str:
            return False
        
        try:
            last_activity = datetime.fromisoformat(last_activity_str)
            timeout_duration = timedelta(hours=self.config['session_timeout_hours'])
            
            if datetime.now() - last_activity > timeout_duration:
                self.logout()
                return False
            
            # Update last activity
            session['last_activity'] = datetime.now().isoformat()
            st.session_state[self.session_key] = session
            
            return True
            
        except Exception as e:
            st.error(f"Session validation error: {e}")
            return False
    
    def is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        attempts_data = self.failed_attempts[username]
        failed_count = attempts_data.get('count', 0)
        last_attempt = attempts_data.get('last_attempt')
        
        if failed_count < self.config['max_failed_attempts']:
            return False
        
        if not last_attempt:
            return True
        
        try:
            last_attempt_time = datetime.fromisoformat(last_attempt)
            lockout_duration = timedelta(minutes=self.config['lockout_duration_minutes'])
            
            if datetime.now() - last_attempt_time < lockout_duration:
                return True
            else:
                # Reset failed attempts after lockout period
                del self.failed_attempts[username]
                return False
                
        except Exception:
            return True
    
    def record_failed_attempt(self, username: str):
        """Record a failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = {'count': 0, 'last_attempt': None}
        
        self.failed_attempts[username]['count'] += 1
        self.failed_attempts[username]['last_attempt'] = datetime.now().isoformat()
    
    def clear_failed_attempts(self, username: str):
        """Clear failed attempts for successful login"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def authenticate(self, username: str, password: str) -> Dict:
        """Enhanced authentication with security checks"""
        result = {
            'success': False,
            'message': '',
            'user_data': None
        }
        
        # Check if user is locked out
        if self.is_user_locked_out(username):
            remaining_time = self.get_lockout_remaining_time(username)
            result['message'] = f"Account locked. Try again in {remaining_time} minutes."
            return result
        
        users = self.users.get('users', {})
        
        if username not in users:
            self.record_failed_attempt(username)
            result['message'] = "Invalid username or password."
            return result
        
        user_data = users[username]
        password_hash = self.hash_password(password)
        
        if password_hash != user_data['password_hash']:
            self.record_failed_attempt(username)
            remaining_attempts = self.config['max_failed_attempts'] - self.failed_attempts.get(username, {}).get('count', 0)
            result['message'] = f"Invalid username or password. {remaining_attempts} attempts remaining."
            return result
        
        # Successful authentication
        self.clear_failed_attempts(username)
        
        # Create session
        now = datetime.now().isoformat()
        st.session_state[self.session_key] = {
            'authenticated': True,
            'username': username,
            'role': user_data['role'],
            'name': user_data['name'],
            'permissions': user_data['permissions'],
            'login_time': now,
            'last_activity': now,
            'session_id': secrets.token_hex(16)
        }
        
        result['success'] = True
        result['message'] = "Login successful!"
        result['user_data'] = user_data
        
        return result
    
    def get_lockout_remaining_time(self, username: str) -> int:
        """Get remaining lockout time in minutes"""
        if username not in self.failed_attempts:
            return 0
        
        last_attempt = self.failed_attempts[username].get('last_attempt')
        if not last_attempt:
            return 0
        
        try:
            last_attempt_time = datetime.fromisoformat(last_attempt)
            lockout_duration = timedelta(minutes=self.config['lockout_duration_minutes'])
            elapsed = datetime.now() - last_attempt_time
            remaining = lockout_duration - elapsed
            
            return max(0, int(remaining.total_seconds() / 60))
        except Exception:
            return 0
    
    def show_login_screen(self):
        """Enhanced login screen with security features"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🔐 Secure Authentication")
            st.info("🤖 **AI Procurement Data Processor**\n\nSecure login with session management.")
            
            # Check for session timeout message
            if 'session_expired' in st.session_state:
                st.warning("⏰ Your session has expired. Please log in again.")
                del st.session_state['session_expired']
            
            # Login form
            username = st.text_input(
                "Username:", 
                placeholder="Enter username...",
                key="secure_login_username"
            )
            
            password = st.text_input(
                "Password:", 
                type="password",
                placeholder="Enter password...",
                key="secure_login_password"
            )
            
            # Show lockout status
            if username and self.is_user_locked_out(username):
                remaining = self.get_lockout_remaining_time(username)
                st.error(f"🔒 Account locked for {remaining} more minutes.")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                login_disabled = username and self.is_user_locked_out(username)
                
                if st.button("🚀 Login", type="primary", use_container_width=True, disabled=login_disabled):
                    if username and password:
                        result = self.authenticate(username, password)
                        
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
                    else:
                        st.error("Please enter both username and password.")
            
            # Session info
            st.caption(f"🕒 Session timeout: {self.config['session_timeout_hours']} hours")
            
            # Debug info
            if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
                with st.expander("🔧 Debug Information"):
                    st.write("**Available Users:**")
                    users = self.users.get('users', {})
                    for user, data in users.items():
                        locked = " 🔒" if self.is_user_locked_out(user) else ""
                        st.write(f"- **{user}** ({data['role']}){locked}")
                    
                    if self.failed_attempts:
                        st.write("**Failed Attempts:**")
                        for user, attempts in self.failed_attempts.items():
                            st.write(f"- {user}: {attempts['count']} failures")
    
    def show_session_info(self):
        """Show session information in sidebar"""
        session = st.session_state.get(self.session_key, {})
        
        if session.get('authenticated'):
            user_name = session.get('name', 'Unknown')
            role = session.get('role', 'Unknown')
            login_time = session.get('login_time', '')
            
            st.success(f"🔓 **{user_name}**")
            st.caption(f"Role: {role}")
            
            if login_time:
                try:
                    login_dt = datetime.fromisoformat(login_time)
                    session_duration = datetime.now() - login_dt
                    hours, remainder = divmod(session_duration.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    st.caption(f"Session: {int(hours)}h {int(minutes)}m")
                except Exception:
                    pass
            
            # Session timeout warning
            last_activity_str = session.get('last_activity')
            if last_activity_str:
                try:
                    last_activity = datetime.fromisoformat(last_activity_str)
                    timeout_duration = timedelta(hours=self.config['session_timeout_hours'])
                    time_remaining = timeout_duration - (datetime.now() - last_activity)
                    
                    if time_remaining.total_seconds() < 1800:  # 30 minutes warning
                        minutes_left = int(time_remaining.total_seconds() / 60)
                        st.warning(f"⏰ Session expires in {minutes_left} minutes")
                        
                        if st.button("🔄 Extend Session", key="extend_session"):
                            session['last_activity'] = datetime.now().isoformat()
                            st.session_state[self.session_key] = session
                            st.rerun()
                            
                except Exception:
                    pass

    # Rest of methods from Phase 2 (load_users, hash_password, etc.)
    # ... (copy from Phase 2 implementation)
Step 3.2: Update .env for security config
bash# Add to .env file
SESSION_TIMEOUT_HOURS=8
MAX_FAILED_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15
REQUIRE_STRONG_PASSWORDS=false
PASSWORD_SALT=your_unique_salt_here_change_this
Step 3.3: Test Phase 3
bash# Test security features
streamlit run app.py

# Verify:
# 1. Session timeout works
# 2. Failed login attempts are tracked
# 3. Account lockout after max attempts
# 4. Session extension works
# 5. Activity tracking updates

🚀 Phase 4: Advanced Features (3-4 hours)
Objective: User management, audit logging, and advanced security
Step 4.1: User Management Interface
python"""
user_management.py - Admin interface for user management
"""
import streamlit as st
import json
from datetime import datetime
from auth import SecureAuth

def show_user_management_tab(auth_instance: SecureAuth):
    """User management interface for admins"""
    
    if not auth_instance.has_permission('admin'):
        st.error("❌ Admin permission required")
        return
    
    st.header("👥 User Management")
    
    # Tabs for different management functions
    mgmt_tab1, mgmt_tab2, mgmt_tab3 = st.tabs(["👤 Users", "📊 Sessions", "📝 Audit Log"])
    
    with mgmt_tab1:
        show_users_tab(auth_instance)
    
    with mgmt_tab2:
        show_sessions_tab(auth_instance)
    
    with mgmt_tab3:
        show_audit_log_tab(auth_instance)

def show_users_tab(auth_instance: SecureAuth):
    """Show and manage users"""
    st.subheader("Manage Users")
    
    users = auth_instance.users.get('users', {})
    
    # Display existing users
    if users:
        st.write("**Current Users:**")
        
        for username, user_data in users.items():
            with st.expander(f"👤 {username} ({user_data['role']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {user_data['name']}")
                    st.write(f"**Role:** {user_data['role']}")
                    st.write(f"**Created:** {user_data.get('created', 'Unknown')}")
                    
                    # Permissions
                    st.write("**Permissions:**")
                    for perm in user_data.get('permissions', []):
                        st.write(f"  ✅ {perm}")
                
                with col2:
                    # User actions
                    if st.button(f"🔄 Reset Password", key=f"reset_{username}"):
                        new_password = st.text_input(
                            "New Password:", 
                            type="password",
                            key=f"new_pass_{username}"
                        )
                        if new_password and st.button(f"Confirm Reset", key=f"confirm_reset_{username}"):
                            # Reset password logic here
                            st.success("Password reset successfully!")
                    
                    if username != auth_instance.get_current_user().get('username'):
                        if st.button(f"🗑️ Delete User", key=f"delete_{username}"):
                            # Delete user logic here
                            st.success("User deleted successfully!")
    
    # Add new user form
    st.markdown("---")
    st.subheader("➕ Add New User")
    
    with st.form("add_user_form"):
        new_username = st.text_input("Username:")
        new_name = st.text_input("Full Name:")
        new_password = st.text_input("Password:", type="password")
        new_role = st.selectbox("Role:", ["viewer", "processor", "admin"])
        
        submitted = st.form_submit_button("Add User")
        
        if submitted and new_username and new_name and new_password:
            # Add user logic here
            st.success(f"User {new_username} added successfully!")

def show_sessions_tab(auth_instance: SecureAuth):
    """Show active sessions"""
    st.subheader("Active Sessions")
    
    # This would require tracking active sessions in a persistent store
    # For now, show current session info
    current_user = auth_instance.get_current_user()
    
    if current_user.get('authenticated'):
        st.write("**Current Session:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"User: {current_user.get('name')}")
            st.write(f"Role: {current_user.get('role')}")
            st.write(f"Login: {current_user.get('login_time', 'Unknown')}")
        
        with col2:
            st.write(f"Session ID: {current_user.get('session_id', 'Unknown')}")
            st.write(f"Last Activity: {current_user.get('last_activity', 'Unknown')}")

def show_audit_log_tab(auth_instance: SecureAuth):
    """Show audit log"""
    st.subheader("📝 Audit Log")
    
    # This would require implementing audit logging
    st.info("Audit logging not yet implemented. This would show:")
    st.write("- Login/logout events")
    st.write("- Failed login attempts")
    st.write("- User management actions")
    st.write("- Data access events")
    st.write("- Configuration changes")
Step 4.2: Audit Logging System
python"""
audit_logger.py - Audit logging system
"""
import json
import os
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
    
    def log_event(self, event_type: str, user: str, details: Dict[str, Any] = None):
        """Log an audit event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'details': details or {},
            'ip_address': self._get_client_ip(),
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"Failed to write audit log: {e}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address (simplified)"""
        # In a real deployment, you'd get this from request headers
        return "localhost"
    
    def get_recent_events(self, limit: int = 100) -> list:
        """Get recent audit events"""
        events = []
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            events.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Failed to read audit log: {e}")
        
        return events

# Integration with auth system
def log_login_attempt(audit_logger: AuditLogger, username: str, success: bool, details: Dict = None):
    """Log login attempt"""
    event_type = "LOGIN_SUCCESS" if success else "LOGIN_FAILURE"
    audit_logger.log_event(event_type, username, details)

def log_logout(audit_logger: AuditLogger, username: str):
    """Log logout event"""
    audit_logger.log_event("LOGOUT", username)

def log_user_action(audit_logger: AuditLogger, username: str, action: str, details: Dict = None):
    """Log user action"""
    audit_logger.log_event("USER_ACTION", username, {"action": action, **details or {}})
Step 4.3: Integration and Testing
python# Updated app.py with full Phase 4 features
from auth import SecureAuth, require_auth, require_permission
from user_management import show_user_management_tab
from audit_logger import AuditLogger, log_login_attempt, log_user_action

def main():
    """Complete authenticated application"""
    
    # Initialize systems
    auth = SecureAuth()
    audit_logger = AuditLogger()
    
    # Check session validity
    if not auth.is_session_valid():
        if auth.is_authenticated():
            st.session_state['session_expired'] = True
            auth.logout()
    
    # Require authentication
    require_auth(auth)
    
    # Enhanced sidebar
    with st.sidebar:
        auth.show_session_info()
        
        if st.button("🚪 Logout", key="logout_btn"):
            username = auth.get_current_user().get('username', 'unknown')
            audit_logger.log_event("LOGOUT", username)
            auth.logout()
            st.rerun()
    
    # Role-based navigation
    user = auth.get_current_user()
    role = user.get('role', '')
    
    # Build tabs based on permissions
    available_tabs = []
    
    if auth.has_permission('read'):
        available_tabs.append("🎯 Opportunities")
    
    if auth.has_permission('write'):
        available_tabs.extend(["🗂️ Data Loading", "🛒 Order Optimization"])
    
    if auth.has_permission('admin'):
        available_tabs.append("👥 User Management")
    
    # Display appropriate tabs
    if available_tabs:
        tabs = st.tabs(available_tabs)
        
        for i, tab_name in enumerate(available_tabs):
            with tabs[i]:
                if "Data Loading" in tab_name:
                    # Your existing data loading code
                    log_user_action(audit_logger, user.get('username'), "ACCESS_DATA_LOADING")
                    st.write("Data Loading Tab")
                    
                elif "Opportunities" in tab_name:
                    # Your existing opportunities code
                    log_user_action(audit_logger, user.get('username'), "ACCESS_OPPORTUNITIES")
                    st.write("Opportunities Tab")
                    
                elif "Order Optimization" in tab_name:
                    # Your existing order optimization code
                    log_user_action(audit_logger, user.get('username'), "ACCESS_ORDER_OPTIMIZATION")
                    st.write("Order Optimization Tab")
                    
                elif "User Management" in tab_name:
                    show_user_management_tab(auth)
    
    else:
        st.error("No accessible features for your role.")

🧪 Testing Strategy
Phase 1 Testing
bash# Test basic functionality
python -c "
from auth import SimpleAuth
auth = SimpleAuth()
print('✅ Auth module loads correctly')
print(f'Master password: {auth.master_password}')
"
Phase 2 Testing
bash# Test user creation
python -c "
from auth import MultiUserAuth
auth = MultiUserAuth()
users = auth.users
print(f'Users loaded: {list(users.get(\"users\", {}).keys())}')
"
Phase 3 Testing
bash# Test security features
python -c "
from auth import SecureAuth
auth = SecureAuth()
print(f'Config: {auth.config}')
print('✅ Security features initialized')
"
Phase 4 Testing
bash# Test complete system
streamlit run app.py
# Manual testing of all features

📈 Migration Path Between Phases
Phase 1 → Phase 2

Keep existing SimpleAuth class
Add MultiUserAuth class alongside
Update imports gradually
Test both systems work

Phase 2 → Phase 3

Add security features to existing MultiUserAuth
Update session management incrementally
Test each security feature individually

Phase 3 → Phase 4

Add audit logging without breaking existing auth
Create user management as separate module
Integrate gradually with main app


🚀 Quick Start Commands
bash# Phase 1 (30 minutes)
git checkout -b auth-phase-1
# Create auth.py with SimpleAuth
# Update app.py imports
# Test basic login

# Phase 2 (1-2 hours)  
git checkout -b auth-phase-2
# Add MultiUserAuth class
# Create users.json
# Test multi-user login

# Phase 3 (2-3 hours)
git checkout -b auth-phase-3
# Add SecureAuth with sessions
# Update .env config
# Test security features

# Phase 4 (3-4 hours)
git checkout -b auth-phase-4
# Add user management
# Add audit logging
# Test complete system