# app.py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
from inference import generate_caption

# ── Checkpoint paths — update these to point to your downloaded .pt files ──
CKPT_PATHS = {
    "bn": "checkpoints/clipcap_bn_best.pt",
    "hi": "checkpoints/clipcap_hi_best.pt",
    "en": "checkpoints/clipcap_en_best.pt",
}

LANG_NAMES = {
    "bn": "Bengali (বাংলা)",
    "hi": "Hindi (हिन्दी)",
    "en": "English",
}

LANG_COLORS = {
    "bn": "#1565C0",
    "hi": "#E65100",
    "en": "#2E7D32",
}


class CaptionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CLIPCap — Trilingual Image Captioning")
        self.root.geometry("900x680")
        self.root.resizable(True, True)
        self.root.configure(bg="#F5F5F5")

        self.image_path: str | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self):
        # ── Title bar ────────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg="#1A237E", pady=10)
        title_frame.pack(fill=tk.X)
        tk.Label(
            title_frame,
            text="CLIPCap — Trilingual Image Captioning",
            font=("Helvetica", 16, "bold"),
            fg="white", bg="#1A237E",
        ).pack()
        tk.Label(
            title_frame,
            text="Bengali  ·  Hindi  ·  English",
            font=("Helvetica", 11),
            fg="#BBDEFB", bg="#1A237E",
        ).pack()

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#F5F5F5", padx=16, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        # Left: image panel
        left = tk.Frame(body, bg="#F5F5F5")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.img_label = tk.Label(
            left,
            text="No image selected\n\nClick 'Upload Image' to begin",
            bg="#E3F2FD", width=42, height=18,
            font=("Helvetica", 11), fg="#90A4AE",
            relief=tk.FLAT, bd=0,
        )
        self.img_label.pack(pady=(0, 10))

        upload_btn = tk.Button(
            left,
            text="📂  Upload Image",
            command=self._upload_image,
            bg="#1A237E", fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, padx=14, pady=8,
            cursor="hand2",
            activebackground="#283593", activeforeground="white",
        )
        upload_btn.pack(fill=tk.X, padx=4)

        # Right: controls + output
        right = tk.Frame(body, bg="#F5F5F5", padx=12)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Language selector
        tk.Label(right, text="Select Caption Language:",
                 font=("Helvetica", 11, "bold"),
                 bg="#F5F5F5", fg="#333333").pack(anchor=tk.W, pady=(0, 4))

        self.lang_var = tk.StringVar(value="en")
        lang_frame = tk.Frame(right, bg="#F5F5F5")
        lang_frame.pack(fill=tk.X, pady=(0, 10))
        for code, name in LANG_NAMES.items():
            col = LANG_COLORS[code]
            rb = tk.Radiobutton(
                lang_frame, text=name, variable=self.lang_var, value=code,
                font=("Helvetica", 11),
                bg="#F5F5F5", fg=col, selectcolor="#E8EAF6",
                activebackground="#F5F5F5",
            )
            rb.pack(anchor=tk.W, padx=8, pady=2)

        # Beam slider
        tk.Label(right, text="Beam Size:",
                 font=("Helvetica", 10, "bold"),
                 bg="#F5F5F5", fg="#555555").pack(anchor=tk.W)
        self.beam_var = tk.IntVar(value=5)
        beam_frame = tk.Frame(right, bg="#F5F5F5")
        beam_frame.pack(fill=tk.X, pady=(0, 10))
        self.beam_slider = tk.Scale(
            beam_frame, from_=1, to=6,
            orient=tk.HORIZONTAL, variable=self.beam_var,
            bg="#F5F5F5", highlightthickness=0,
            font=("Helvetica", 9),
        )
        self.beam_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(beam_frame, textvariable=self.beam_var,
                 font=("Helvetica", 10, "bold"),
                 bg="#F5F5F5", fg="#1A237E", width=3).pack(side=tk.LEFT)

        # Generate button
        self.gen_btn = tk.Button(
            right,
            text="⚡  Generate Caption",
            command=self._generate,
            bg="#2E7D32", fg="white",
            font=("Helvetica", 12, "bold"),
            relief=tk.FLAT, padx=14, pady=10,
            cursor="hand2",
            activebackground="#388E3C", activeforeground="white",
            state=tk.DISABLED,
        )
        self.gen_btn.pack(fill=tk.X, pady=(0, 12), padx=4)

        # Output display
        tk.Label(right, text="Generated Caption:",
                 font=("Helvetica", 11, "bold"),
                 bg="#F5F5F5", fg="#333333").pack(anchor=tk.W)

        out_frame = tk.Frame(right, bg="#F5F5F5")
        out_frame.pack(fill=tk.BOTH, expand=True)

        self.out_text = tk.Text(
            out_frame,
            height=6, wrap=tk.WORD,
            font=("Noto Sans", 13),
            bg="white", fg="#1A237E",
            relief=tk.FLAT, bd=1,
            padx=10, pady=8,
            state=tk.DISABLED,
        )
        self.out_text.pack(fill=tk.BOTH, expand=True)

        # Copy button
        copy_btn = tk.Button(
            right,
            text="📋  Copy Caption",
            command=self._copy_caption,
            bg="#546E7A", fg="white",
            font=("Helvetica", 10),
            relief=tk.FLAT, padx=10, pady=5,
            cursor="hand2",
            activebackground="#607D8B", activeforeground="white",
        )
        copy_btn.pack(fill=tk.X, padx=4, pady=(6, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready — upload an image to begin.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 9), fg="#666666",
            bg="#EEEEEE", anchor=tk.W, padx=10, pady=4,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        try:
            img = Image.open(path).convert("RGB")
            # Fit into display box (max 400×320)
            img.thumbnail((400, 320), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.configure(
                image=self.photo, text="", bg="white",
                width=self.photo.width(), height=self.photo.height(),
            )
            self.gen_btn.configure(state=tk.NORMAL)
            self.status_var.set(f"Image loaded: {os.path.basename(path)}")
            self._set_output("")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")

    def _generate(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        lang = self.lang_var.get()
        ckpt = CKPT_PATHS.get(lang)
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror(
                "Checkpoint Missing",
                f"Checkpoint not found:\n{ckpt}\n\nUpdate CKPT_PATHS in app.py.",
            )
            return
        self.gen_btn.configure(state=tk.DISABLED, text="⏳  Generating...")
        self.status_var.set(f"Generating {LANG_NAMES[lang]} caption ...")
        self._set_output("")

        def _worker():
            try:
                caption = generate_caption(
                    image_path=self.image_path,
                    lang=lang,
                    ckpt_path=ckpt,
                    beam=self.beam_var.get(),
                )
                self.root.after(0, lambda: self._on_done(caption, lang))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, caption: str, lang: str):
        self._set_output(caption)
        self.gen_btn.configure(state=tk.NORMAL, text="⚡  Generate Caption")
        self.status_var.set(
            f"Caption generated in {LANG_NAMES[lang]}.  "
            f"({len(caption.split())} words)"
        )

    def _on_error(self, msg: str):
        self.gen_btn.configure(state=tk.NORMAL, text="⚡  Generate Caption")
        self.status_var.set("Error during generation.")
        messagebox.showerror("Generation Error", msg)

    def _set_output(self, text: str):
        self.out_text.configure(state=tk.NORMAL)
        self.out_text.delete("1.0", tk.END)
        if text:
            self.out_text.insert(tk.END, text)
        self.out_text.configure(state=tk.DISABLED)

    def _copy_caption(self):
        text = self.out_text.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("Caption copied to clipboard.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CaptionApp(root)
    root.mainloop()