"""ค่าคงที่กลาง: path ของไฟล์ข้อมูลและโฟลเดอร์ผลลัพธ์"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # โฟลเดอร์ GA-Sport-Day
DEFAULT_CSV = BASE_DIR / "data" / "HR.csv"
OUTPUT_DIR = BASE_DIR / "output"
