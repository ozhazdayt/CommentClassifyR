import tkinter as tk
from tkinter import scrolledtext
# Yorum ön işleme ve model yükleme kodları burada olacak

def classify_comment():
    user_comment = comment_entry.get("1.0", tk.END)
    preprocessed_comment = preprocess_text_turkish(user_comment)
    # Sınıflandırma ve sonuçları gösterme işlemleri burada

root = tk.Tk()
root.title("Yorum Sınıflandırıcı")

# Yorum giriş alanı
comment_label = tk.Label(root, text="Yorum:")
comment_label.pack()
comment_entry = scrolledtext.ScrolledText(root, height=5)
comment_entry.pack()

# Sınıflandırma butonu
classify_btn = tk.Button(root, text="Sınıflandır", command=classify_comment)
classify_btn.pack()

# Sonuç alanı
result_label = tk.Label(root, text="Sınıflandırma Sonuçları:")
result_label.pack()
result_display = tk.Label(root, text="", fg="blue")
result_display.pack()

root.mainloop()
