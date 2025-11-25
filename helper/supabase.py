import os
from typing import List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("❌ Missing Supabase environment variables!")
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")

# Create and export the Supabase client
supabase: Client = create_client(url, key)

print("✅ Supabase client initialized")

def get_user_inventory(user_id: int) -> List[Dict[str, Any]]:
    """
    Fetches the inventory for a specific user from the 'inventory' table.
    
    Args:
        user_id (int): The foreign key ID of the user.
        
    Returns:
        List[Dict[str, Any]]: A list of inventory items (dictionaries).
    """
    try:
        # Query the 'inventory' table: select all columns where user_id matches
        response = supabase.table("inventory").select("*").eq("user_id", user_id).execute()
        
        # Return the list of data (rows)
        return response.data
    except Exception as e:
        print(f"❌ Error fetching inventory: {e}")
        return []