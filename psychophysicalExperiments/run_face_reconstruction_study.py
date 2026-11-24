"""Run the local, 50-trial face reconstruction slideshow."""
import argparse
from pathlib import Path
import secrets
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk
if not __package__:
    # Direct launch from the study folder also needs the shared renderer.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.face_recipes_and_observations import FACTOR_NAMES, FACTOR_VALUES
from psychophysicalExperiments.face_study_trials_and_scoring import (
    Session, DEFAULT_SESSIONS, DISPLAY_ORDER, POSTER, RESIZE, trial_cues, hint_text,
    observation, option_icon, display_name, pixels, DEFAULT_VISIBILITY_COUNTS,
    VISIBILITY_LABELS, NOISE,
)


class Slideshow:
    def __init__(self, window, sessions=DEFAULT_SESSIONS, seed=None, cued=True,
                 hint_range=(0, 6), hint_accuracy=None, show_counts=True,
                 visibility_counts=DEFAULT_VISIBILITY_COUNTS, noise_cap=max(NOISE)):
        self.window, self.sessions, self.seed, self.cued = window, sessions, seed, cued
        self.hint_range, self.hint_accuracy, self.show_counts = hint_range, hint_accuracy, show_counts
        self.visibility_counts, self.noise_cap = visibility_counts, noise_cap
        self.session = None
        self.last_size = None
        self.resize_job = None
        window.title("Face reconstruction")
        window.geometry(f"{int(.94*window.winfo_screenwidth())}x{int(.88*window.winfo_screenheight())}")
        if window.tk.call("tk", "windowingsystem") == "win32" and window.state() != "withdrawn":
            window.state("zoomed")
        window.minsize(580, 440)
        window.protocol("WM_DELETE_WINDOW", self.quit)
        style = self.style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white", foreground=POSTER["ink"], font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Accent.TButton", background=POSTER["teal"], foreground="white", padding=(18, 7))
        style.map("Accent.TButton", background=[("active", "#005753"), ("disabled", "#CED8D8")])
        self.canvas = tk.Canvas(window, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        for axis, orient, command in ((0, "vertical", self.canvas.yview),(1, "horizontal", self.canvas.xview)):
            bar = ttk.Scrollbar(window, orient=orient, command=command)
            bar.grid(row=axis, column=1-axis, sticky="ns" if axis==0 else "ew")
            self.canvas.configure(**{("yscrollcommand" if axis==0 else "xscrollcommand"): bar.set})
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        self.page = ttk.Frame(self.canvas, padding=(24, 16))
        self.item = self.canvas.create_window(0, 0, window=self.page, anchor="nw")
        self.canvas.bind("<Configure>", self.layout)
        self.page.bind("<Configure>", self.layout)
        window.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-int(e.delta/120), "units"))
        self.setup()

    def layout(self, event=None):
        self.canvas.itemconfigure(self.item, width=max(self.canvas.winfo_width(), self.page.winfo_reqwidth()))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if event and event.widget == self.canvas:
            if self.resize_job:
                self.window.after_cancel(self.resize_job)
            self.resize_job = self.window.after_idle(self.fit_page)

    def fit_page(self):
        """Scale to the actual viewport, including when a participant resizes mid-trial."""
        self.resize_job = None
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1 or (width, height) == self.last_size:
            return
        self.last_size = width, height
        # Smaller windows can scroll the minimum usable layout.
        width, height = max(830, width), max(610, height)
        text = max(15, min(25, round(min(width/73, height/43))))
        family = "Segoe UI"
        self.style.configure("TLabel", font=(family, -text))
        self.style.configure("Header.TLabel", font=(family, -round(1.65*text), "bold"))
        self.style.configure("Feature.TLabel", font=(family, -text, "bold"))
        self.style.configure("Visibility.TLabel", font=(family, -text, "bold"), foreground=POSTER["teal"])
        for element in ("TButton", "Accent.TButton", "TCheckbutton", "TEntry", "TSpinbox", "TCombobox"):
            self.style.configure(element, font=(family, -text))
        margin = max(16, int(width*.023))
        self.page.configure(padding=(margin, round(text*.7)))
        inner = width-2*margin
        self.instructions.configure(wraplength=inner-12)
        # Reserve the measured headings and the action button, including padding.
        available = max(350, height-self.instructions.winfo_reqheight()-self.title.winfo_reqheight()-5*text-64)
        if self.slide_image:
            label, source = self.slide_image
            ratio = min((inner-16)/source.width, available/source.height)
            self.set_image(label, source.resize((round(source.width*ratio), round(source.height*ratio)),
                                                Image.Resampling.LANCZOS))
        if self.evidence is None:
            return
        evidence_label, source = self.evidence
        side = max(200, int(min(inner*.345, available*.77)))
        self.set_image(evidence_label, source.resize((side, side), RESIZE))
        if self.hints is not None:
            self.hints.configure(wraplength=side)
        # Give the layout row more drawing area. Its examples contain a whole face.
        art_height = max(35, int((available-6*(text+18))/6.7))
        art_width = max(110, int(inner*.145))
        for j in DISPLAY_ORDER:
            icon_height = round(art_height * (1.7 if FACTOR_NAMES[j] == "layout" else 1))
            for k in range(3):
                button = self.option_buttons[3*j+k]
                button.configure(font=(family, -text, "underline" if self.trial_hints[j] == k else "normal"))
                self.set_image(button, option_icon(j, k, (art_width, icon_height)))

    def clear(self):
        for child in self.page.winfo_children():
            child.destroy()
        self.last_size = self.slide_image = self.evidence = self.hints = None
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def heading(self, title, text):
        self.title = ttk.Label(self.page, text=title, style="Header.TLabel")
        self.title.pack(anchor="w")
        self.instructions = ttk.Label(self.page, text=text, wraplength=900, justify="left")
        self.instructions.pack(anchor="w", pady=(4,12))

    def set_image(self, widget, img):
        widget.picture = ImageTk.PhotoImage(img, master=self.window)
        widget.configure(image=widget.picture)

    def action(self, label, command):
        button = ttk.Button(self.page, text=label, style="Accent.TButton", command=command)
        button.pack(anchor="e", pady=8)
        return button

    def setup(self):
        self.clear()
        self.heading("Face reconstruction", "Enter your name and session number to begin.")
        ttk.Label(self.page,text="Participant name").pack(anchor="w")
        self.name = ttk.Entry(self.page,width=32)
        self.name.pack(anchor="w", pady=(2,12))
        self.name.focus_set()
        ttk.Label(self.page,text="Session number").pack(anchor="w")
        self.number = ttk.Entry(self.page,width=12)
        self.number.pack(anchor="w")
        self.number.insert(0,"1")
        self.setup_visibility()
        if self.cued:
            self.setup_hints()
        self.action("Start session", self.start)
        self.fit_page()

    def setup_visibility(self):
        settings = ttk.Frame(self.page)
        expanded = tk.BooleanVar(master=self.window, value=False)

        def toggle():
            if expanded.get():
                settings.pack(anchor="w", after=button)
            else:
                settings.pack_forget()

        button = ttk.Checkbutton(self.page, text="Configure visibility and noise",
                                 variable=expanded, command=toggle)
        button.pack(anchor="w", pady=(12, 3))
        self.visibility_inputs = [tk.StringVar(master=self.window, value=str(count))
                                  for count in self.visibility_counts]
        ttk.Label(settings, text="Enter 50 trials across the four visible-pixel bands.").grid(
            row=0, column=0, columnspan=2, sticky="w")
        for row, (label, value) in enumerate(zip(VISIBILITY_LABELS, self.visibility_inputs), 1):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w")
            ttk.Spinbox(settings, from_=0, to=50, textvariable=value, width=5).grid(row=row, column=1, padx=6)
        self.max_noise = tk.StringVar(master=self.window, value=f"{self.noise_cap:.0%}")
        ttk.Label(settings, text="Maximum pixel-flip probability").grid(row=5, column=0, sticky="w")
        ttk.Combobox(settings, textvariable=self.max_noise, width=6, state="readonly",
                     values=[f"{noise:.0%}" for noise in NOISE]).grid(row=5, column=1, padx=6)

    def setup_hints(self):
        settings = ttk.Frame(self.page)
        expanded = tk.BooleanVar(master=self.window, value=False)
        def toggle():
            if expanded.get():
                settings.pack(anchor="w", after=toggle_button)
            else:
                settings.pack_forget()
        toggle_button = ttk.Checkbutton(self.page, text="Configure session hints",
                                        variable=expanded, command=toggle)
        toggle_button.pack(anchor="w", pady=(12, 3))
        self.hint_min = tk.StringVar(master=self.window, value=str(self.hint_range[0]))
        self.hint_max = tk.StringVar(master=self.window, value=str(self.hint_range[1]))
        self.hint_chance = tk.StringVar(master=self.window, value="" if self.hint_accuracy is None else str(self.hint_accuracy))
        self.disclose = tk.BooleanVar(master=self.window, value=self.show_counts)
        for i, (label, value) in enumerate((("Minimum hints per trial", self.hint_min),
                                            ("Maximum hints per trial", self.hint_max))):
            ttk.Label(settings, text=label).grid(row=i, column=0, sticky="w")
            ttk.Spinbox(settings, from_=0, to=6, textvariable=value, width=5).grid(row=i, column=1, padx=6)
        ttk.Label(settings, text="Chance a hint is correct (0–1)").grid(row=2, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.hint_chance, width=7).grid(row=2, column=1)
        ttk.Label(settings, text="Leave blank to sample every possible correct count equally.").grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(settings, text="Show correct and incorrect totals",
                        variable=self.disclose).grid(row=4, column=0, columnspan=2, sticky="w")

    def start(self):
        try:
            seed = self.seed if self.seed is not None else secrets.randbits(48)
            if self.cued:
                self.hint_range = (int(self.hint_min.get()), int(self.hint_max.get()))
                chance = self.hint_chance.get().strip()
                self.hint_accuracy = float(chance) if chance else None
                self.show_counts = self.disclose.get()
            visibility_counts = tuple(int(value.get()) for value in self.visibility_inputs)
            noise_cap = int(self.max_noise.get().rstrip("%")) / 100
            self.session = Session(self.sessions, self.name.get(), self.number.get(), seed, self.cued,
                                   self.hint_range, self.hint_accuracy, self.show_counts,
                                   visibility_counts, noise_cap)
        except FileExistsError:
            messagebox.showerror("Session already exists", "Choose a new session number. The existing session is saved.")
            return
        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot start session", str(exc))
            return
        self.intro(False)

    def saved_image(self, filename):
        with Image.open(self.session.folder/filename) as img:
            label = ttk.Label(self.page)
            label.pack(pady=2)
            self.slide_image = label, img.copy()
        self.fit_page()

    def intro(self, typical):
        self.clear()
        if typical:
            instructions = "Choose each feature according to the face you think was originally shown."
            if self.cued:
                instructions += " Hints can be wrong."
                if self.show_counts:
                    instructions += " You will see how many hints are correct and incorrect."
            self.heading("Typical faces", "These are common combinations. Any combination is possible. "
                         + instructions)
            self.saved_image("typical_faces.png")
            self.action("Begin 50 trials", self.trial)
        else:
            self.heading("Learn the six features", "Each row shows the three choices for one feature. "
                         "Eye spacing changes the distance between the eyes. Layout changes the placement of the features.")
            self.saved_image("feature_guide.png")
            self.action("Next", lambda: self.intro(True))

    def trial(self):
        self.clear()
        trial = self.session.plan[len(self.session.rows)]
        hint_instruction = "An underlined option is the hint for that feature. Some hints are incorrect. " if self.cued else ""
        self.heading(f"Trial {trial['trial']} of 50", "Choose one option in every row to recover the original face. "
                     + hint_instruction + "Blue-gray blocks are hidden. Visible black and white pixels can contain noise. "
                     "Feel free to take an educated guess when too little of the face is visible to decide. "
                     "Keep in mind the typical combinations you saw at the beginning.")
        body = ttk.Frame(self.page)
        body.pack(fill="x")
        body.columnconfigure(0, weight=4, uniform="panel")
        body.columnconfigure(1, weight=6, uniform="panel")
        left = ttk.Frame(body)
        left.grid(row=0,column=0,sticky="n",padx=(0,18))
        observed, mask = observation(self.session.lib, trial)
        evidence_label = ttk.Label(left)
        evidence_label.pack()
        self.evidence = evidence_label, pixels(observed, mask)
        ttk.Label(left, text=f"{100*mask.mean():.1f}% visible", style="Visibility.TLabel").pack(
            anchor="w", pady=(8, 0))
        state = self.session.lib.states[trial["state_id"]]
        cues = trial_cues(state, trial)
        self.trial_hints = [-1] * len(FACTOR_NAMES) if cues is None else cues
        if self.cued:
            hints = hint_text(cues, trial["correct_cues"], trial["hint_counts_shown"])
            self.hints = ttk.Label(left, text=hints, wraplength=240, justify="left")
            self.hints.pack(anchor="w", pady=(10, 2))
        choices = ttk.Frame(body)
        choices.grid(row=0,column=1,sticky="new")
        for col in range(1, 4):
            choices.columnconfigure(col, weight=1, uniform="options")
        self.values = [tk.IntVar(master=self.window, value=-1) for _ in FACTOR_NAMES]
        self.option_buttons = [None] * 18
        for row, j in enumerate(DISPLAY_ORDER):
            ttk.Label(choices,text=FACTOR_NAMES[j].capitalize(),style="Feature.TLabel").grid(
                row=row,column=0,sticky="w",padx=(0,12))
            for k, value in enumerate(FACTOR_VALUES[j]):
                button = tk.Radiobutton(choices,text=display_name(value), variable=self.values[j],
                    value=k, compound="top", indicatoron=False,
                    font=("Segoe UI",10,"underline" if self.trial_hints[j] == k else "normal"),
                    foreground=POSTER["ink"], background="white",
                    selectcolor="#B6DFDB", activebackground="#D7EBE9", relief="ridge",
                    borderwidth=2, command=self.check, padx=4,pady=2)
                button.grid(row=row,column=k+1,padx=3,pady=2,sticky="nsew")
                self.option_buttons[3*j+k] = button
        self.confirm = self.action("Confirm recipe", self.submit)
        self.check()
        self.fit_page()
        self.started = time.perf_counter()

    def check(self):
        self.confirm.state(["!disabled"] if all(v.get()>=0 for v in self.values) else ["disabled"])

    def submit(self):
        if self.confirm.instate(["disabled"]):
            return
        self.confirm.state(["disabled"])
        try:
            filename = self.session.record([v.get() for v in self.values], time.perf_counter()-self.started)
        except OSError as exc:
            messagebox.showerror("Response could not be saved", str(exc))
            self.check()
            return
        self.clear()
        self.heading("Trial feedback", "Compare your reconstruction with the original face.")
        self.saved_image(filename)
        self.action("Finish session" if len(self.session.rows)==50 else "Next trial",
                    self.finish if len(self.session.rows)==50 else self.trial)

    def finish(self):
        self.clear()
        self.heading("Session complete", "All 50 responses and feedback images have been saved.")
        from psychophysicalExperiments.analyze_face_reconstruction_accuracy import contour
        y = "correct_cues" if any(row["cues_given"] for row in self.session.rows) else "noise_pct"
        path = self.session.folder/"session_accuracy.png"
        contour(self.session.rows,"actual_visibility_pct",y,path,"Participant recipe accuracy")
        self.saved_image(path.name)
        self.action("Close", self.window.destroy)

    def quit(self):
        if self.session and len(self.session.rows)<50 and not messagebox.askyesno(
                "End session?", f"{len(self.session.rows)} confirmed responses have been saved. End this session now?"):
            return
        self.window.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible session seed")
    parser.add_argument("--uncued", action="store_true", help="Show only the masked image and choices")
    parser.add_argument("--hint-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=(0, 6),
                        help="Range of hints per trial, inclusive (default 0 6)")
    parser.add_argument("--hint-accuracy", type=float, help="Probability per hint of being correct. "
                        "By default all possible correct counts are equally likely for a given total.")
    parser.add_argument("--hide-hint-counts", action="store_true", help="Hide the aggregate correct/incorrect counts")
    parser.add_argument("--visibility-counts", type=int, nargs=4, metavar=("UNDER20","20_TO_35","35_TO_60","60_PLUS"),
                        default=DEFAULT_VISIBILITY_COUNTS, help="Number of trials in each fixed visibility band (must total 50)")
    parser.add_argument("--noise-cap", type=int, choices=[round(100*n) for n in NOISE], default=20,
                        help="Maximum pixel-flip probability in percent")
    args = parser.parse_args()
    window = tk.Tk()
    app = Slideshow(window, args.sessions, args.seed, not args.uncued, args.hint_range,
                    args.hint_accuracy, not args.hide_hint_counts, args.visibility_counts, args.noise_cap/100)
    window.mainloop()
