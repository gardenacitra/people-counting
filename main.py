import tkinter as tk
import os

from tkinter import ttk

def open_new_window():
    new_windows = tk.Toplevel(windows)
    new_windows.geometry("500x500")
    new_windows.title("About Us")

    # Menambahkan label tentang aplikasi dengan wraplength yang cukup besar supaya tidak terpotong
    about_label = ttk.Label(new_windows,
                            text='Aplikasi ini merupakan aplikasi yang digunakan untuk mendeteksi dan menghitung '
                                 'jumlah jemaat Gereja Tiberias Boston Square Cibubur',
                            font=('Helvetica', 12, 'bold'),
                            wraplength=400,
                            justify='center')

    # Menambahkan padding di sekeliling label
    about_label.pack(pady=15, padx=10)

    close_button = ttk.Button(new_windows, text='Close', command=new_windows.destroy)
    close_button.pack(side='bottom', pady=80)

    # Mengatur posisi label agar berada di tengah-tengah window
    window_width = new_windows.winfo_reqwidth()
    window_height = new_windows.winfo_reqheight()

    x_pos = int((new_windows.winfo_screenwidth() / 2) - (window_width / 2))
    y_pos = int((new_windows.winfo_screenheight() / 2) - (window_height / 2))

    new_windows.geometry("+{}+{}".format(x_pos, y_pos))

    about_label.place(relx=.5,rely=.4,anchor="center")

# Fungsi yang akan dipanggil ketika tombol 1 diklik
def button1_clicked():
    print("Tombol 1 diklik")
    os.chdir(r"C:\Users\taruna.ersa\PycharmProjects\counting-and-tracking")
    os.system("python main2.py --weights yolo.pt --no-trace --view-img --source coba.mp4 --classes 0")

# Fungsi yang akan dipanggil ketika tombol 2 diklik
def button2_clicked():
    print("Tombol 2 diklik")
    open_new_window()

# Membuat jendela utama
windows = tk.Tk()
windows.geometry("750x750")
windows.title("People Counting")

# Membuat frame input
input_frame = ttk.Frame(windows)
input_frame.pack(padx=10, pady=10, fill="x", expand=True)

# Menambahkan judul aplikasi
post = ttk.Label(input_frame, text="Aplikasi Pendeteksi dan Penghitung Objek Manusia", font=("Helvetica", 16, "bold"))
post.pack(padx=10, pady=10, fill="x", expand=True)
post.configure(anchor="center")

# Menambahkan keterangan lokasi
post1 = ttk.Label(input_frame, text="Gereja Tiberias Boston Square Cibubur", font=("Helvetica", 14))
post1.pack(padx=10, pady=10, fill="x", expand=True)
post1.configure(anchor="center")

# Membuat frame untuk tombol
button_frame = ttk.Frame(input_frame)
button_frame.pack(padx=10, pady=40)

# Membuat gaya untuk tombol
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12, "bold"), foreground="white", background="#3498db")

# Membuat tombol 1
button1 = ttk.Button(button_frame, text="Start", command=button1_clicked, style="TButton")
button1.pack(side="left", padx=5)

# Membuat tombol 2
button2 = ttk.Button(button_frame, text="About", command=button2_clicked, style="TButton")
button2.pack(side="left", padx=5)

windows.mainloop()