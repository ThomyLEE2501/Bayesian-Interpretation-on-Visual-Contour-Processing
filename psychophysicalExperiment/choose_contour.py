"""Choose CSV data, axes, filters and an output PNG in a local desktop window."""
import csv
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psychophysicalExperiment.face_accuracy import AXES, FIGURES, EXAMPLE_PRIOR_WEIGHT, contour


def main():
    window = tk.Tk()
    window.title("Recipe accuracy plots")
    frame = ttk.Frame(window, padding=20)
    frame.pack()
    loaded = []
    status = tk.StringVar()
    x, y = tk.StringVar(), tk.StringVar()
    where = tk.StringVar()
    log_visibility = tk.BooleanVar(value=False)
    title = tk.StringVar(value="Face recipe accuracy")

    def load(paths):
        if not paths:
            return
        candidate = []
        try:
            for path in paths:
                with open(path, encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if "score" not in (reader.fieldnames or []):
                        raise ValueError(f"{Path(path).name} has no six-feature score column.")
                    candidate.extend(reader)
            available = [k for k in AXES if any(r.get(k) not in ("", None) for r in candidate)]
            if len(available)<2:
                raise ValueError("These files need two measured axes.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot read data", str(exc))
            return
        loaded[:] = candidate
        for box in boxes:
            box.configure(values=available)
        x.set("visibility_pct" if "visibility_pct" in available else available[0])
        y.set(next(k for k in ("correct_cues","noise_pct",*available) if k in available and k!=x.get()))
        where.set("")
        status.set(f"{len(loaded):,} responses from " + ", ".join(Path(p).name for p in paths))

    def export():
        if not loaded:
            messagebox.showinfo("Choose data", "Select a session responses.csv or model_scores.csv.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG figure","*.png")])
        if path:
            try:
                filters = dict(s.strip().split("=", 1) for s in where.get().split(",") if s.strip())
                contour(loaded, x.get(), y.get(), path, title.get(), filters, log_visibility=log_visibility.get())
                messagebox.showinfo("Plot saved", path)
            except (ValueError, OSError) as exc:
                messagebox.showerror("Cannot plot selection", str(exc))

    ttk.Button(frame, text="Choose one or more CSV files", command=lambda: load(
        filedialog.askopenfilenames(filetypes=[("Response or model data","*.csv")]))).grid(row=0,column=0,columnspan=2,sticky="w")
    ttk.Label(frame,textvariable=status,wraplength=500).grid(row=1,column=0,columnspan=2,sticky="w",pady=6)
    boxes=[]
    for i, (label,variable) in enumerate((("Horizontal axis",x),("Vertical axis",y)), start=2):
        ttk.Label(frame,text=label).grid(row=i,column=0,sticky="w")
        box=ttk.Combobox(frame,textvariable=variable,state="readonly",width=36)
        box.grid(row=i,column=1,pady=3)
        boxes.append(box)
    for i,(label,var) in enumerate((("Plot title",title),("Optional filters",where)),start=4):
        ttk.Label(frame,text=label).grid(row=i,column=0,sticky="w")
        ttk.Entry(frame,textvariable=var,width=40).grid(row=i,column=1,pady=4)
    ttk.Label(frame,text="Example filters: cues_given=4,noise_pct=5").grid(row=6,column=0,columnspan=2,sticky="w")
    ttk.Checkbutton(frame, text="Logarithmic visibility axis", variable=log_visibility).grid(
        row=7, column=0, columnspan=2, sticky="w")
    ttk.Button(frame,text="Save contour PNG",command=export).grid(row=8,column=1,sticky="e",pady=8)
    default = FIGURES/"model_scores.csv"
    if default.exists():
        load([default])
        where.set(f"prior_weight={EXAMPLE_PRIOR_WEIGHT:g},noise_pct=5")
    window.mainloop()


if __name__=="__main__":
    main()
