import csv
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, PhotoImage
from PIL import Image, ImageTk  # Import Pillow library for image handling

def load_log_file(log_filepath):
    """
    Loads guiding log data from a .log file.

    Args:
        log_filepath (str): The full path to the .log file.

    Returns:
        str or dict: String containing the log content if successful,
                     or a dictionary with an error message if loading fails.
    """
    try:
        with open(log_filepath, 'r') as log_file:
            log_content = log_file.read()
        return log_content
    except FileNotFoundError:
        return {"error": f"File not found: {log_filepath}"}
    except Exception as e:
        return {"error": f"Error loading log file: {e}"}


def analyse_guiding_log(log_content):
    """
    Analyses astronomy guiding log data to calculate RMS, detect oscillation,
    and suggest corrections.

    Args:
        log_content (str): A string containing the guiding log data.

    Returns:
        dict: A dictionary containing analysis results, including RMS values,
              oscillation info, and correction suggestions.
    """
    log_data = StringIO(log_content)
    reader = csv.reader(log_data)
    headers = next(reader, None)  # Skip header line

    if headers is None:
        return {"error": "No data found in log content."}

    data_start_line = -1
    for i, row in enumerate(reader):
        if row and row[0].strip() == '"Timestamp': # Find line with actual data headers if file has extra preamble.
            data_start_line = i
            headers = row
            break
        if i > 10: # Limit search in first 10 lines, if not found assume data starts after first line.
            break

    log_data = StringIO(log_content) # Reset to start
    reader = csv.reader(log_data)

    if data_start_line != -1: # Skip lines before data start
        for _ in range(data_start_line +1):
            next(reader)
    else:
         headers = next(reader, None) # If no "Timestamp" header found, assume first line is header


    ra_diff_arcsec = []
    dec_diff_arcsec = []
    timestamps = []

    try:
        for row in reader:
            if not row:  # Skip empty lines
                continue
            try:
                timestamp_str = row[0].strip().strip('"') # Remove quotes
                ra_diff = float(row[5].strip().strip('"')) # RA Dif(")
                dec_diff = float(row[6].strip().strip('"')) # Dec Dif(")

                timestamps.append(timestamp_str)
                ra_diff_arcsec.append(ra_diff)
                dec_diff_arcsec.append(dec_diff)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping line due to parsing error: {row} - {e}") # Still print to console for debugging
                continue

    except csv.Error as e:
        return {"error": f"CSV parsing error: {e}"}

    if not ra_diff_arcsec or not dec_diff_arcsec:
        return {"error": "No valid data columns (RA Dif(\"), Dec Dif(\")) found in log."}

    ra_diff_arcsec_np = np.array(ra_diff_arcsec)
    dec_diff_arcsec_np = np.array(dec_diff_arcsec)

    rms_ra = np.sqrt(np.mean(ra_diff_arcsec_np**2))
    rms_dec = np.sqrt(np.mean(dec_diff_arcsec_np**2))
    total_rms = np.sqrt(rms_ra**2 + rms_dec**2) # Combined RMS

    analysis_results = {
        "rms_ra": rms_ra,
        "rms_dec": rms_dec,
        "total_rms": total_rms,
        "timestamps": timestamps,
        "ra_diff_arcsec": ra_diff_arcsec,
        "dec_diff_arcsec": dec_diff_arcsec,
    }

    # Oscillation detection (basic sign change method)
    oscillation_ra = detect_oscillation(ra_diff_arcsec_np)
    oscillation_dec = detect_oscillation(dec_diff_arcsec_np)
    analysis_results["oscillation_ra"] = oscillation_ra
    analysis_results["oscillation_dec"] = oscillation_dec

    # Suggest corrections
    analysis_results["suggestions"] = suggest_corrections(analysis_results)

    return analysis_results


def detect_oscillation(data, threshold_sign_changes=0.3): # Threshold as ratio of sign changes to data points
    """
    Detects oscillation in the guiding data based on sign changes.
    A very basic method. More advanced methods could use frequency analysis.

    Args:
        data (np.array): 1D numpy array of guiding error data (RA or Dec diff).
        threshold_sign_changes (float): Ratio of sign changes to data points to flag as oscillation.

    Returns:
        dict: Dictionary indicating if oscillation is detected and sign change ratio.
    """
    sign_changes = 0
    for i in range(1, len(data)):
        if (data[i] >= 0 and data[i-1] < 0) or (data[i] < 0 and data[i-1] >= 0):
            sign_changes += 1

    sign_change_ratio = sign_changes / len(data) if len(data) > 0 else 0
    is_oscillating = sign_change_ratio > threshold_sign_changes

    return {
        "is_oscillating": is_oscillating,
        "sign_change_ratio": sign_change_ratio
    }


def suggest_corrections(analysis_results):
    """
    Suggests corrections based on the analysis results.

    Args:
        analysis_results (dict): Dictionary from analyse_guiding_log.

    Returns:
        list: List of suggestion strings.
    """
    suggestions = []
    if "error" in analysis_results:
        return [analysis_results["error"]]

    if analysis_results["rms_ra"] > 1.0 or analysis_results["rms_dec"] > 1.0: # Example threshold, adjust as needed
        suggestions.append("High RMS detected. Consider these potential issues:")
        if analysis_results["rms_ra"] > 1.0:
            suggestions.append(f"  - RA RMS is high ({analysis_results['rms_ra']:.2f} arcsec).")
        if analysis_results["rms_dec"] > 1.0:
            suggestions.append(f"  - Dec RMS is high ({analysis_results['rms_dec']:.2f} arcsec).")
        suggestions.append("  - Check seeing conditions and focus.")
        suggestions.append("  - Ensure mount is balanced in RA and Dec.")
        suggestions.append("  - Verify polar alignment accuracy.")
        suggestions.append("  - Check for mechanical issues in the mount (backlash, stiction).")


    if analysis_results["oscillation_ra"]["is_oscillating"] or analysis_results["oscillation_dec"]["is_oscillating"]:
        suggestions.append("Oscillation detected in guiding:")
        if analysis_results["oscillation_ra"]["is_oscillating"]:
            suggestions.append(f"  - RA axis oscillation detected (Sign change ratio: {analysis_results['oscillation_ra']['sign_change_ratio']:.2f}).")
        if analysis_results["oscillation_dec"]["is_oscillating"]:
            suggestions.append(f"  - Dec axis oscillation detected (Sign change ratio: {analysis_results['oscillation_dec']['sign_change_ratio']:.2f}).")

        suggestions.append("  - Reduce guiding aggressiveness (reduce RA/Dec Aggression or similar parameter).")
        suggestions.append("  - Adjust hysteresis or backlash settings in your guiding software.")
        suggestions.append("  - Ensure guide camera and scope are securely mounted and minimize flexure.")
        suggestions.append("  - If using PHD2, consider running the Guiding Assistant for recommendations.")


    if not suggestions:
        suggestions.append("Guiding performance seems good based on RMS and oscillation analysis.")
        suggestions.append("However, always review the graphs for visual inspection of guiding behavior.")

    return suggestions


def plot_guiding_data(analysis_results, output_path="guiding_plot.png"):
    """
    Plots the guiding data (RA and Dec Diff) over time and returns the filepath.

    Args:
        analysis_results (dict): Dictionary from analyse_guiding_log.
        output_path (str, optional): Path to save the plot image. Defaults to "guiding_plot.png".

    Returns:
        str or None: Path to the saved plot image if successful, None if error.
    """
    if "error" in analysis_results:
        print("Error in analysis, cannot plot.") # Still print to console for debugging
        return None # Indicate plot failed

    timestamps = analysis_results["timestamps"]
    ra_diff_arcsec = analysis_results["ra_diff_arcsec"]
    dec_diff_arcsec = analysis_results["dec_diff_arcsec"]

    # Using index as x if timestamps are problematic to parse, otherwise parse timestamps to datetime and use.
    x_axis = range(len(timestamps)) # or  pd.to_datetime(timestamps) if you parse timestamps


    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, ra_diff_arcsec, label='RA Diff (")', alpha=0.7)
    plt.plot(x_axis, dec_diff_arcsec, label='Dec Diff (")', alpha=0.7)

    plt.xlabel('Time (Sample Index)') # or 'Time' if timestamps parsed
    plt.ylabel('Guiding Error (arcseconds)')
    plt.title('Astronomy Guiding Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() # Close plot after saving, don't display inline in GUI for now.
    print(f"Guiding data plot saved to {output_path}") # Still print to console for debugging
    return output_path # Return path to saved plot


class GuidingAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Guiding Log Analyzer")

        self.log_content = None # Store loaded log content here
        self.analysis_results = None # Store analysis results
        self.plot_filepath = None # Store saved plot path
        self.tk_plot_image = None # To hold PhotoImage to prevent garbage collection

        self.load_button = tk.Button(master, text="Load Log File", command=self.load_and_analyze_log)
        self.load_button.pack(pady=10)

        self.plot_image_label = tk.Label(master) # Label to display the plot image
        self.plot_image_label.pack(pady=10, padx=10)

        self.result_text = scrolledtext.ScrolledText(master, height=10, width=80) # Reduced height for text area
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED) # Make it read-only


    def load_and_analyze_log(self):
        log_filepath = filedialog.askopenfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if log_filepath:
            self.result_text.config(state=tk.NORMAL) # Enable editing temporarily
            self.result_text.delete(1.0, tk.END) # Clear previous text
            self.result_text.insert(tk.END, "Loading log file...\n")
            self.result_text.config(state=tk.DISABLED) # Disable editing again
            self.plot_image_label.config(image=None) # Clear previous plot

            log_content_or_error = load_log_file(log_filepath)

            if isinstance(log_content_or_error, dict) and "error" in log_content_or_error:
                error_message = f"Error loading log file: {log_content_or_error['error']}"
                self.display_results_text(error_message)
                self.log_content = None # Reset
                self.analysis_results = None # Reset
                self.plot_filepath = None # Reset
                self.tk_plot_image = None # Reset
            else:
                self.log_content = log_content_or_error
                self.analyze_and_display()


    def analyze_and_display(self):
        if self.log_content:
            self.result_text.config(state=tk.NORMAL) # Enable editing temporarily
            self.result_text.delete(1.0, tk.END) # Clear previous text
            self.result_text.insert(tk.END, "Analyzing guiding log...\n")
            self.result_text.config(state=tk.DISABLED) # Disable editing again

            self.analysis_results = analyse_guiding_log(self.log_content)

            if "error" in self.analysis_results:
                error_message = f"Error during analysis: {self.analysis_results['error']}"
                self.display_results_text(error_message)
                self.plot_filepath = None # Reset
                self.tk_plot_image = None # Reset
                self.plot_image_label.config(image=None) # Clear plot in GUI in case of error
            else:
                results_text = "Guiding Analysis Results:\n"
                results_text += f"  RA RMS: {self.analysis_results['rms_ra']:.3f} arcsec\n"
                results_text += f"  Dec RMS: {self.analysis_results['rms_dec']:.3f} arcsec\n"
                results_text += f"  Total RMS: {self.analysis_results['total_rms']:.3f} arcsec\n"
                results_text += f"  RA Oscillation Detected: {self.analysis_results['oscillation_ra']['is_oscillating']} (Sign Change Ratio: {self.analysis_results['oscillation_ra']['sign_change_ratio']:.2f})\n"
                results_text += f"  Dec Oscillation Detected: {self.analysis_results['oscillation_dec']['is_oscillating']} (Sign Change Ratio: {self.analysis_results['oscillation_dec']['sign_change_ratio']:.2f})\n\n"
                results_text += "Correction Suggestions:\n"
                for suggestion in self.analysis_results['suggestions']:
                    results_text += f"  - {suggestion}\n"

                self.display_results_text(results_text)
                self.plot_filepath = plot_guiding_data(self.analysis_results) # Save plot and get path

                if self.plot_filepath:
                    # Load the saved plot image and display it in the GUI
                    image = Image.open(self.plot_filepath)
                    image = image.resize((600, 300), Image.LANCZOS) # Resize image to fit in GUI - adjust size as needed
                    self.tk_plot_image = ImageTk.PhotoImage(image) # Keep reference to PhotoImage
                    self.plot_image_label.config(image=self.tk_plot_image) # Set image to label
                    self.plot_label.config(text=f"Guiding plot displayed above and saved to: {self.plot_filepath}") # Update label text
                else:
                     self.plot_label.config(text="Error generating plot.") # Indicate plot error in GUI
                     self.plot_image_label.config(image=None) # Clear any previous plot image


        else:
            self.display_results_text("No log file loaded yet.")
            self.plot_filepath = None # Reset plot path if no log loaded
            self.plot_image_label.config(image=None) # Clear plot if no log loaded
            self.tk_plot_image = None # Clear image reference


    def display_results_text(self, text):
        self.result_text.config(state=tk.NORMAL) # Enable editing to insert text
        self.result_text.delete(1.0, tk.END) # Clear previous text
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED) # Disable editing again


if __name__ == "__main__":
    root = tk.Tk()
    gui = GuidingAnalyzerGUI(root)
    root.mainloop()
