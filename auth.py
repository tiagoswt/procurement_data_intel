"""
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
        self.master_password = os.getenv("MASTER_PASSWORD", "procurement2024")
        self.session_key = "auth_simple"

    def hash_password(self, password: str) -> str:
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return st.session_state.get(self.session_key, {}).get("authenticated", False)

    def authenticate(self, password: str) -> bool:
        """Authenticate with master password"""
        if password == self.master_password:
            st.session_state[self.session_key] = {
                "authenticated": True,
                "login_time": st.session_state.get("current_time", "unknown"),
                "user": "admin",
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
            st.info(
                "🤖 **AI Procurement Data Processor**\n\nEnter the master password to access the system."
            )

            # Login form
            password = st.text_input(
                "Master Password:",
                type="password",
                placeholder="Enter password...",
                key="login_password",
            )

            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 Login", type="primary", width='stretch'):
                    if self.authenticate(password):
                        st.success("✅ Authentication successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid password. Please try again.")

            # Development info (remove in production)
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                st.caption(f"Debug: Master password is '{self.master_password}'")


def require_auth(auth_instance: SimpleAuth):
    """Decorator function to require authentication"""
    if not auth_instance.is_authenticated():
        auth_instance.show_login_screen()
        st.stop()
