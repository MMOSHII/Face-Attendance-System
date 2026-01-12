import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys, os
import pandas as pd
from PIL import Image, ImageTk

try:
    from config import LOG_PATH, IMAGES_DIR, LOGS_DIR
    from data_manager import load_data, load_attendance, save_data
    from student_ops import add_student, edit_student, delete_student
    from face_utils import train_model
    from logger import log_message
    from exceptions import set_log_box
except ImportError:
    from utils.config import LOG_PATH, IMAGES_DIR, LOGS_DIR
    from utils.data_manager import load_data, load_attendance, save_data
    from utils.student_ops import add_student, edit_student, delete_student
    from utils.face_utils import train_model
    from utils.logger import log_message
    from utils.exceptions import set_log_box


def search_dataframe(df, query: str):
    if not query:
        return df
    query = query.lower()
    mask = df.astype(str).agg(lambda row: row.str.lower().str.contains(query).any(), axis=1)
    return df[mask]


class StudentManagerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Student Data Manager with Face Recognition [DEBUG MODE]")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f6fa")

        self.student_df = load_data()
        self.attendance_df = load_attendance()

        self.entries = {}
        self.label_widgets = {}
        self.buttons = {}
        self.image_label = None
        self.tree = None
        self.history_tree = None
        self.log_box = None
        self.search_entry = None
        self.notebook = None

        self.build_menus()
        self.build_form()
        self.build_actions()
        self.build_search()
        self.build_notebook()
        self.build_log()

        self.refresh_treeview(self.tree, self.student_df)
        self.refresh_treeview(self.history_tree, self.attendance_df)
        log_message("🚀 Program started", self.log_box)
        set_log_box(self.log_box)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_menus(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Data", command=lambda: save_data(self.student_df), accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        log_menu = tk.Menu(menubar, tearoff=0)
        log_menu.add_command(label="Open Log File", command=lambda: Path(LOG_PATH).open())
        log_menu.add_command(label="Clear Log", command=lambda: Path(LOG_PATH).write_text(""))
        menubar.add_cascade(label="Logs", menu=log_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-s>", lambda e: save_data(self.student_df))
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    def build_form(self):
        form_frame = tk.LabelFrame(
            self.root, text="Student Information", padx=15, pady=15,
            bg="#f5f6fa", font=("Arial", 12, "bold")
        )
        form_frame.place(x=20, y=20, width=400, height=350)

        labels = ["Nama", "Kelas", "Total Kehadiran", "Email", "Nomor Telepon", "Waktu Kehadiran"]
        for i, text in enumerate(labels):
            lbl = tk.Label(form_frame, text=text + ":", anchor="w", bg="#f5f6fa", font=("Arial", 10))
            lbl.grid(row=i, column=0, sticky="w", pady=5)
            e = tk.Entry(form_frame, width=30, font=("Arial", 10))
            e.grid(row=i, column=1, pady=5, padx=10, sticky="w")
            self.entries[text] = e
            self.label_widgets[text] = lbl

        self.form_buttons = {}

        clear_btn = tk.Button(
            form_frame, text="✖ Clear Form", command=self.clear_entries,
            bg="#f77862", fg="white", font=("Arial", 10, "bold"), relief="flat"
        )
        clear_btn.grid(row=7, column=1, padx=100, pady=8)
        self.form_buttons["clear"] = clear_btn

        folder_btn = tk.Button(
            form_frame, text="📋 Folder Photo", command=self.open_student_folder,
            bg="#1e8449", fg="white", font=("Arial", 10, "bold"), relief="flat"
        )
        folder_btn.grid(row=7, column=0, padx=10, pady=8)
        self.form_buttons["folder"] = folder_btn

        self.image_label = tk.Label(form_frame, bg="#dcdde1")
        self.image_label.grid(row=0, column=2, rowspan=8, padx=10, pady=5)
        self.image_label.grid_remove()
    
    def open_student_folder(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select a student first!")
            return

        student_id = self.tree.item(selected)["values"][0]
        folder = Path(IMAGES_DIR) / str(student_id)

        if not folder.exists():
            messagebox.showinfo("Info", f"No folder found for student {student_id}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {e}")

    def build_actions(self):
        btn_frame = tk.LabelFrame(
            self.root, text="Actions", padx=10, pady=10,
            bg="#f5f6fa", font=("Arial", 12, "bold")
        )
        btn_frame.place(x=20, y=380, width=400, height=150)

        specs = {
            "add": ("➕ Add (Select Photos)", lambda: add_student(self), "#3498db"),
            "edit": ("✏️ Edit", lambda: edit_student(self), "#f39c12"),
            "delete": ("🗑 Delete", lambda: delete_student(self), "#e74c3c"),
            "train": ("🧠 Train Model", lambda: train_model(self.log_box), "#27ae60"),
        }

        for i, (key, (text, cmd, bg)) in enumerate(specs.items()):
            row, col = divmod(i, 2)
            btn = tk.Button(
                btn_frame, text=text, command=cmd,
                bg=bg, fg="white", font=("Arial", 10, "bold"),
                relief="flat", padx=10, pady=5
            )
            btn.grid(row=row, column=col, padx=10, pady=8, sticky="ew")
            self.buttons[key] = btn

    def build_search(self):
        search_frame = tk.LabelFrame(
            self.root, text="Search Student", padx=10, pady=10,
            bg="#f5f6fa", font=("Arial", 12, "bold")
        )
        search_frame.place(x=450, y=20, width=720, height=80)

        tk.Label(search_frame, text="Search:", font=("Arial", 10), bg="#f5f6fa").pack(side="left", padx=5)
        self.search_entry = tk.Entry(search_frame, width=40, font=("Arial", 10))
        self.search_entry.pack(side="left", padx=5)

        tk.Button(search_frame, text="🔍 Search", command=self.global_search,
                  bg="#2980b9", fg="white", font=("Arial", 9, "bold"),
                  relief="flat").pack(side="left", padx=5)

        tk.Button(search_frame, text="✖ Clear", command=self.global_clear,
                  bg="#7f8c8d", fg="white", font=("Arial", 9, "bold"),
                  relief="flat").pack(side="left", padx=5)
        
        tk.Button(search_frame, text="Export", command=self.global_export,
                  bg="#2f9459", fg="white", font=("Arial", 9, "bold"),
                  relief="flat").pack(side="left", padx=5)

    def build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.place(x=450, y=120, width=720, height=411)

        student_tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(student_tab, text="📋 Students")

        cols = ["ID", "Nama", "Kelas", "Total Kehadiran", "Email", "Nomor Telepon", "Waktu Kehadiran"]
        self.tree = ttk.Treeview(student_tab, columns=cols, show="headings", height=12)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        history_tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(history_tab, text="🕒 Attendance History")

        hist_cols = ["id", "name", "date", "status"]
        self.history_tree = ttk.Treeview(history_tab, columns=hist_cols, show="headings", height=12)
        for col in hist_cols:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, anchor="center", stretch=True)
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<<TreeviewSelect>>", self.on_history_select)

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def build_log(self):
        log_frame = tk.LabelFrame(
            self.root, text="System Log", padx=5, pady=5,
            bg="#f5f6fa", font=("Arial", 12, "bold")
        )
        log_frame.place(x=20, y=540, width=1150, height=140)

        self.log_box = tk.Text(log_frame, height=6, wrap="word", bg="black", fg="lime", font=("Consolas", 10))
        self.log_box.pack(fill="both", expand=True)

    def refresh_treeview(self, tree, df):
        tree.delete(*tree.get_children())
        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def clear_entries(self):
        for e in self.entries.values():
            e.delete(0, tk.END)

    def set_entries(self, values):
        self.clear_entries()
        for key, v in zip(self.entries.keys(), values[1:]):
            self.entries[key].insert(0, v)

    def on_select(self, event):
        selected = self.tree.selection()
        if selected:
            self.set_entries(self.tree.item(selected)["values"])

    def on_history_select(self, event):
        selected = self.history_tree.selection()
        if not selected:
            return

        values = self.history_tree.item(selected)["values"]
        student_id, date = values[0], values[2]
        filename = f"{student_id}-{date.replace('-','').replace(':','').replace(' ','')}.png"
        img_path = Path(LOGS_DIR) / filename

        if not img_path.exists():
            log_message(f"❌ Image not found: {img_path}", self.log_box)
            return

        try:
            img = Image.open(img_path).resize((300, 300))
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.image_label.grid()

            for widget in self.entries.values():
                widget.grid_remove()
            for widget in self.label_widgets.values():
                widget.grid_remove()
            for btn in self.form_buttons.values():
                btn.grid_remove()

        except Exception as e:
            log_message(f"❌ Failed to load image {filename}: {e}", self.log_box)

    def on_tab_change(self, event):
        tab = self.notebook.tab(self.notebook.select(), "text")
        log_message(f"📑 Switched to {tab} tab", self.log_box)

        if tab == "📋 Students":
            self.image_label.grid_remove()
            for lbl in self.label_widgets.values():
                lbl.grid()
            for entry in self.entries.values():
                entry.grid()
            for btn in self.form_buttons.values():
                btn.grid()
            self.refresh_treeview(self.tree, self.student_df)
            for btn in ["add", "edit", "delete"]:
                self.buttons[btn].config(state="normal")
        else:
            self.refresh_treeview(self.history_tree, self.attendance_df)
            for btn in ["add", "edit", "delete"]:
                self.buttons[btn].config(state="disabled")

    def global_search(self):
        query = self.search_entry.get().strip().lower()
        tab = self.notebook.tab(self.notebook.select(), "text")
        if tab == "📋 Students":
            filtered = search_dataframe(self.student_df, query)
            self.refresh_treeview(self.tree, filtered)
        else:
            filtered = search_dataframe(self.attendance_df, query)
            self.refresh_treeview(self.history_tree, filtered)
        log_message(f"🔍 Found {len(filtered)} result(s) for '{query}'", self.log_box)

    def global_clear(self):
        tab = self.notebook.tab(self.notebook.select(), "text")
        if tab == "📋 Students":
            self.refresh_treeview(self.tree, self.student_df)
        else:
            self.refresh_treeview(self.history_tree, self.attendance_df)

    def global_export(self):
        filepath = r"Data\attendance_history.csv"

        required_columns = {
            'id', 'name', 'timestamp', 'status'
        }

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
        except Exception as e:
            raise Exception(f"Gagal membaca CSV: {e}")

        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Kolom berikut hilang: {', '.join(missing)}")

        df['id'] = df['id'].astype(str)
        df['name'] = df['name'].astype(str)
        df['timestamp'] = df['timestamp'].astype(str)
        df['status'] = df['status'].astype(str)

        downloads_path = str(Path.home() / "Downloads")

        save_path = os.path.join(downloads_path, "attendance_export.csv")
        df.to_csv(save_path, index=False)

    def select_photos(self):
        filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png *.jpeg")])

    def on_close(self):
        log_message("🛑 Program closed", self.log_box)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StudentManagerApp()
    app.run()