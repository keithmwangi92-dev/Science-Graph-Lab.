import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "Graphs_Output")
os.makedirs(SAVE_FOLDER, exist_ok=True)

def open_save_folder():
    folder = os.path.abspath(SAVE_FOLDER)
    print(f"\n=== DEBUG ===")
    print(f"Looking in: {folder}")
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"Files found ({len(files)}): {files}")
    else:
        print("Folder does NOT exist!")
    print("==============\n")
    try:
        if sys.platform == "win32":
            os.startfile(folder)
        elif "microsoft" in open("/proc/version").read().lower() if os.path.exists("/proc/version") else "":
            # WSL - open in Windows Explorer
            subprocess.Popen(["explorer.exe", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as e:
        print(f"Error opening folder: {e}")
        print(f"Manually open: {folder}")

# ================= ALL FORMULAS EVER TAUGHT =================
def f_linear(x,m,c): return m*x + c
def f_parabola(x,a,b,c): return a*x**2 + b*x + c
def f_cubic(x,a,b,c,d): return a*x**3 + b*x**2 + c*x + d
def f_exponential(x,a,b): return a*np.exp(b*x)
def f_exp_decay(x,a,b,c): return a*np.exp(-b*x) + c
def f_power(x,a,b): return a*np.power(np.abs(x)+1e-9, b)
def f_log(x,a,b): return a*np.log(np.abs(x)+1e-9) + b
def f_sine(x,A,B,C,D): return A*np.sin(B*x + C) + D
def f_cosine(x,A,B,C,D): return A*np.cos(B*x + C) + D
def f_hooke(x,k): return k*x
def f_inverse(x,a): return a/x

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else -1

FORMULAS = {
    "Hooke's Law (F=kx) origin": f_hooke,
    "Linear y=mx+c": f_linear,
    "Parabola y=ax2+bx+c": f_parabola,
    "Cubic y=ax3+bx2+cx+d": f_cubic,
    "Exponential Growth y=a*e^(bx)": f_exponential,
    "Exponential Decay": f_exp_decay,
    "Power Law y=a*x^b": f_power,
    "Logarithmic y=a*ln(x)+b": f_log,
    "Sine Wave (Waves/SHM)": f_sine,
    "Cosine Wave": f_cosine,
    "Inverse y=a/x": f_inverse,
}

# ================= CORE AUTO BRAIN =================
def analyze_and_plot(x, y, title, x_label="X", y_label="Y"):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    # Remove NaN
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask_valid], y[mask_valid]
    order = np.argsort(x); x, y = x[order], y[order]

    if len(x) < 4:
        print("Need at least 4 points!"); return None

    title_low = title.lower()
    is_likely_hooke = "hooke" in title_low or ("force" in title_low and "extension" in title_low)

    # For Hooke, use only linear part to calculate k
    y_threshold = y.max()*0.85
    mask_linear = y <= y_threshold
    if np.sum(mask_linear) < 3: mask_linear = np.ones_like(y, dtype=bool)
    x_lin, y_lin = x[mask_linear], y[mask_linear]

    best_r2 = -1e9; best_name=""; best_func=None; best_params=None

    for name, func in FORMULAS.items():
        try:
            xt, yt = (x_lin, y_lin) if "Hooke" in name else (x, y)
            # Prevent log(0) / 1/0 issues - filter BOTH x and y together
            if "Log" in name or "Power" in name or "Inverse" in name:
                mask_pos = xt > 0
                xt = xt[mask_pos]; yt = yt[mask_pos]
                if len(xt) < 4: continue
            popt, _ = curve_fit(func, xt, yt, maxfev=50000)
            pred = func(xt, *popt)
            score = r2_score(yt, pred)
            # Boost Hooke if title says Hooke
            if is_likely_hooke and "Hooke" in name: score += 0.15
            if score > best_r2 and np.isfinite(score):
                best_r2, best_name, best_func, best_params = score, name, func, popt
        except Exception:
            continue

    if best_func is None:
        print("Could not fit any formula!"); return None

    # --- PLOTTING ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(11,6.8))
    ax.scatter(x, y, s=90, color='black', edgecolors='white', linewidths=1.2, label='Experimental Data', zorder=6)

    x_smooth = np.linspace(x.min(), x.max()*1.15, 600)
    x_smooth0 = np.linspace(0, x.max()*1.15, 600)

    equation_text = ""

    if "Hooke" in best_name:
        k = best_params[0]
        ax.plot(x_smooth0, best_func(x_smooth0, *best_params), 'r-', linewidth=3.2,
                label=f'IDEAL: F = {k:.3f} x  [k={k:.3f} N/m]', zorder=4)
        # Real curve showing elastic limit
        try:
            spl = make_interp_spline(x, y, k=2)
            y_spl = spl(x_smooth)
            ax.plot(x_smooth, y_spl, color='#1f77b4', alpha=0.5, linewidth=3, label='Actual Behaviour (shows elastic limit & plastic region)', zorder=3)
            ax.axvline(x_lin.max(), color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Limit of Proportionality ({x_lin.max():.2f})')
            ax.axvspan(x_lin.max(), x.max()*1.15, color='orange', alpha=0.08, label='Plastic Region')
        except:
            ax.plot(x, y, color='#1f77b4', alpha=0.5, linewidth=2)
        ax.set_xlim(0, x.max()*1.15); ax.set_ylim(0, y.max()*1.18)
        equation_text = f"F = {k:.4f} * x"
    else:
        # Normal science graphs
        y_smooth = best_func(x_smooth, *best_params)
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label=f'AUTO: {best_name} (R2={best_r2:.4f})', zorder=4)
        # Equation builder
        if "Linear" in best_name: equation_text = f"y = {best_params[0]:.4f}x + {best_params[1]:.4f}"
        elif "Parabola" in best_name: equation_text = f"y = {best_params[0]:.4f}x2 + {best_params[1]:.4f}x + {best_params[2]:.4f}"
        elif "Exponential Growth" in best_name: equation_text = f"y = {best_params[0]:.4f} e^({best_params[1]:.4f}x)"
        elif "Sine" in best_name: equation_text = f"y = {best_params[0]:.3f} sin({best_params[1]:.3f}x + {best_params[2]:.3f}) + {best_params[3]:.3f}"
        elif "Cosine" in best_name: equation_text = f"y = {best_params[0]:.3f} cos({best_params[1]:.3f}x + {best_params[2]:.3f}) + {best_params[3]:.3f}"
        else: equation_text = f"{best_name} Params={np.round(best_params,4)}"

    try:
        area = np.trapezoid(y, x)
    except AttributeError:
        area = np.trapz(y, x)
    ax.set_title(f"{title}\n{best_name} | {equation_text} | R2={best_r2:.4f} | Area={area:.3f}", fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel(x_label, fontsize=12); ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=10, loc='best')

    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    png_path = os.path.join(SAVE_FOLDER, f"{safe_title}_MASTER.png")
    print(f"Attempting to save to: {png_path}")
    plt.tight_layout(); plt.savefig(png_path, dpi=300); plt.close()

    # Excel report
    xlsx_path = png_path.replace('_MASTER.png','_REPORT.xlsx')
    with pd.ExcelWriter(xlsx_path) as w:
        pd.DataFrame({x_label:x, y_label:y}).to_excel(w, sheet_name='Data', index=False)
        pd.DataFrame({
            'Detected Shape':[best_name], 'Equation':[equation_text],
            'R2_Score':[best_r2], 'Area_Trapz':[area], 'Params':[str(best_params)]
        }).to_excel(w, sheet_name='Analysis', index=False)

    print(f"\n{'='*60}\nAUTO-BRAIN RESULT\n{'='*60}")
    print(f"Detected: {best_name}\nEquation: {equation_text}\nR2 = {best_r2:.5f} (1.0 = perfect)\nArea = {area:.4f}")
    print(f"Graph -> {png_path}\nReport -> {xlsx_path}\n{'='*60}\n")
    return png_path

# ================= OTHER TOOLS =================
def load_excel_flow():
    path = input("File path (Excel/CSV) [no quotes]: ").strip().replace('"','').replace("'","")

    if not os.path.exists(path):
        print("File not found!")
        return

    try:
        df = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
    except Exception as e:
        print(f"Read error: {e}")
        return

    print(f"\nColumns found: {list(df.columns)}")
    print(df.head(5).to_string())

    x_col = input(f"X column name [{df.columns[0]}]: ").strip() or df.columns[0]
    y_col = input(f"Y column name [{df.columns[1] if len(df.columns)>1 else df.columns[0]}]: ").strip() or (df.columns[1] if len(df.columns)>1 else df.columns[0])
    title = input(f"Graph Title [{y_col} vs {x_col}]: ").strip() or f"{y_col} vs {x_col}"
    try:
        analyze_and_plot(df[x_col].values, df[y_col].values, title, x_label=x_col, y_label=y_col)
    except Exception as e:
        print(f"Plot failed: {e}")

def manual_flow():
    title = input("Graph Title: ").strip() or "Manual Graph"
    x_label = input("X label: ").strip() or "X"
    y_label = input("Y label: ").strip() or "Y"
    print("Enter X values comma separated: e.g. 0,1,2,3,4")
    x = np.array([float(v) for v in input("X: ").split(',') if v.strip()!=''])
    y = np.array([float(v) for v in input("Y: ").split(',') if v.strip()!=''])
    analyze_and_plot(x,y,title,x_label,y_label)

def projectile_animation():
    try:
        v = float(input("Initial Velocity m/s: "))
        ang = float(input("Angle degrees: "))
        g=9.81
        t_flight = 2*v*np.sin(np.radians(ang))/g
        t = np.linspace(0, t_flight, 300)
        x = v*np.cos(np.radians(ang))*t
        y = v*np.sin(np.radians(ang))*t - 0.5*g*t**2
        analyze_and_plot(x,y, f"Projectile v={v} angle={ang}", "Distance (m)", "Height (m)")
    except Exception as e: print(f"Error: {e}")

def build_powerpoint():
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError:
        print("ERROR: python-pptx not installed. Run: pip install python-pptx")
        return

    folder = os.path.abspath(SAVE_FOLDER)
    images = [f for f in os.listdir(folder) if f.endswith("_MASTER.png")]

    if not images:
        print(f"No graphs found in {folder} - run option 1 or 2 first!")
        return

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Lab Results - All Graphs"
    try:
        slide.shapes.title.text_frame.paragraphs[0].runs[0].font.size = Pt(32)
    except:
        pass

    for img_name in images:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = img_name.replace("_MASTER.png","").replace("_"," ")
        img_path = os.path.join(folder, img_name)
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1), Inches(12), Inches(6))

    ppt_path = os.path.join(folder, "FINAL_PRESENTATION.pptx")
    prs.save(ppt_path)
    print(f"[OK] PowerPoint Saved -> {ppt_path}")

    # Open it
    try:
        if sys.platform == "win32":
            os.startfile(ppt_path)
        else:
            subprocess.Popen(["explorer.exe", ppt_path])
    except:
        pass

# ================= MENU =================
def main():
    print("""
╔════════════════════════════════════════════╗
║  SCIENCE GRAPH LAB                         ║
║  Physics | Chem | Bio | Maths              ║
╚════════════════════════════════════════════╝
    """)
    while True:
        print("\n1. Manual data entry (any graph)")
        print("2. Load Excel/CSV data (AUTO-DETECT shape) ")
        print("3. Projectile Motion")
        print("4. Build PowerPoint from all graphs")
        print("5. Exit (click to automatically open Graphs_Output folder)")
        c = input("Choose [1-5]: ").strip()
        if c=='1': manual_flow()
        elif c=='2': load_excel_flow()
        elif c=='3': projectile_animation()
        elif c=='4': build_powerpoint()
        elif c=='5':
            print("DONE! Opening Graphs_Output/")
            open_save_folder()
            break
        else: print("Invalid!")

if __name__ == "__main__": main()