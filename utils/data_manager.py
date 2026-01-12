import pandas as pd, os
from datetime import datetime, timedelta

try:
    from config import CSV_PATH, ATTENDANCE_PATH
    from logger import log_message
except ImportError:
    from utils.config import CSV_PATH, ATTENDANCE_PATH
    from utils.logger import log_message


def get_next_id(df):
    return 100000 if df.empty else int(df["id"].max()) + 1


# ========================== API ==========================


def update_attendance_record(student: dict, start_time_str: str, end_time_str: str, minutes=10):
    # Parse jam mulai & jam selesai
    START_TIME = datetime.strptime(start_time_str, "%H:%M").time()
    END_TIME = datetime.strptime(end_time_str, "%H:%M").time()

    # Ambil waktu absensi terakhir
    last_time_str = student.get('waktu_kehadiran', None)
    last_time = None
    if last_time_str:
        try:
            last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Tangani format yang tidak cocok
            last_time = None

    now = datetime.now()

    # Cek apakah jam saat ini berada dalam rentang
    if not (START_TIME <= now.time() <= END_TIME):
        return False, now

    # Cegah spam absensi dalam jangka pendek
    if last_time and (now - last_time) < timedelta(minutes=minutes):
        return False, now

    # Update total
    total = student.get('total_kehadiran', 0)
    try:
        total = int(total)
    except ValueError:
        total = 0

    student['total_kehadiran'] = str(total + 1)
    student['waktu_kehadiran'] = now.strftime("%Y-%m-%d %H:%M:%S")

    # Simpan ke database
    save_attendance(student)

    return True, now

def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["id","name","kelas", "total_kehadiran", "email", "nomor_telepon","waktu_kehadiran"])
    return df

def load_attendance():
    if os.path.exists(ATTENDANCE_PATH):
        return pd.read_csv(ATTENDANCE_PATH)
    return pd.DataFrame(columns=["id","name","date","status"])

def save_data(df):
    df.to_csv(CSV_PATH, index=False)
    log_message("✅ Data saved")

def load_students():
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    return {str(row['id']): row.to_dict() for _, row in df.iterrows()}

def save_attendance(student):
    new_row = pd.DataFrame([{
        "id": student['id'],
        "name": student['nama'],
        "timestamp": student['waktu_kehadiran'],
        "status": "Present"
    }])

    new_row.to_csv(ATTENDANCE_PATH, mode='a', index=False, header=False)
    log_message(f"✅ Attendance saved for {student['nama']}")

def save_students(students):
    if not students:
        return
    df = pd.DataFrame(students.values())
    df.to_csv(CSV_PATH, index=False, encoding='utf-8')

def add_student_row(df, entries):
    student_id = get_next_id(df)
    new_row = {
        "id": student_id,
        "nama": entries["Nama"].get(),
        "kelas": entries["Kelas"].get(),
        "total_kehadiran": entries["Total Kehadiran"].get(),
        "email": entries["Email"].get(),
        "nomor_telepon": entries["Nomor Telepon"].get(),
        "waktu_kehadiran": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df, student_id