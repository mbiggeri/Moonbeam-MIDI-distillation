import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import subprocess
import threading
import time
import pygame

# --- MAIN APPLICATION CLASS ---
class MidiGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- PATH CONFIGURATION ---
        # Configure the necessary paths for your project here.
        self.paths = {
            "ckpt_dir": "C:\\Users\\Michael\\Desktop\\ModelliMusicGenerator\\MOONBEAM\\base_model\\moonbeam_839M.pt",
            "tokenizer_path": "tokenizer.model",
            "model_config_path": "C:\\Users\\Michael\\Desktop\\Moonbeam-MIDI-Distillation\\src\\llama_recipes\\configs\\model_config_commu_con_gen.json",
            "additional_token_dict_path": "C:\\Users\\Michael\\Desktop\\MusicDatasets\\Datasets\\ComMU\\processed\\indexed_tokens_dict.json",
            "chord_dict_path": "C:\\Users\\Michael\\Desktop\\MusicDatasets\\Datasets\\ComMU\\processed\\chord_dictionary.json",
            "finetuned_PEFT_weight_path": "C:\\Users\\Michael\\Desktop\\ModelliMusicGenerator\\MOONBEAM\\peft_model",
            "output_dir": "C:\\Users\\Michael\\Desktop\\Generazioni_Moonbeam",
            "project_src_path": "C:\\Users\\Michael\\Desktop\\Moonbeam-MIDI-Distillation\\src",
            "script_to_run": "C:\\Users\\Michael\\Desktop\\Moonbeam-MIDI-Distillation\\recipes\\inference\\custom_music_generation\\generate_single_midi.py"
        }

        self.last_generated_file = None
        
        # --- WINDOW SETUP ---
        self.title("AI MIDI Generator")
        self.geometry("1100x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        self.controls_frame = ctk.CTkScrollableFrame(self.main_frame, label_text="Parameters")
        self.controls_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self.output_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.output_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.output_frame.grid_rowconfigure(1, weight=1)
        
        # --- WIDGETS AND LOGIC ---
        self.widgets = {}
        self.create_control_widgets()
        self.create_output_widgets()
        self.load_and_populate_metadata()

        # Initialize Pygame Mixer for playback
        try:
            pygame.mixer.init()
        except pygame.error as e:
            messagebox.showerror("Playback Error", f"Could not initialize audio playback (Pygame): {e}\nPlayback will be disabled.")


    def create_control_widgets(self):
        """Creates all the input widgets for the generator."""
        # --- Composition ---
        comp_frame = ctk.CTkFrame(self.controls_frame)
        comp_frame.pack(expand=True, fill="x", padx=10, pady=10)
        comp_frame.grid_columnconfigure((0, 1), weight=1)
        
        ctk.CTkLabel(comp_frame, text="Composition", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        self.add_control("audio_key", "Audio Key", comp_frame, "combobox", 1)
        self.add_control("genre", "Genre", comp_frame, "combobox", 1, col=1)
        self.add_control("time_signature", "Time Signature", comp_frame, "combobox", 2)
        self.add_control("num_measures", "Number of Measures", comp_frame, "combobox", 2, col=1)
        self.add_control("bpm", "BPM", comp_frame, "slider", 3, columnspan=2, slider_range=(30, 180), default=120)
        self.add_control("chords", "Chord Progression (optional)", comp_frame, "entry", 4, columnspan=2, placeholder="e.g., Cmaj7 G7 Dm Am")
        
        # --- Instrumentation ---
        inst_frame = ctk.CTkFrame(self.controls_frame)
        inst_frame.pack(expand=True, fill="x", padx=10, pady=10)
        inst_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(inst_frame, text="Instrumentation", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        self.add_control("inst", "Instrument", inst_frame, "combobox", 1)
        self.add_control("track_role", "Track Role", inst_frame, "combobox", 1, col=1)
        self.add_control("pitch_range", "Pitch Range", inst_frame, "combobox", 2)
        self.add_control("sample_rhythm", "Rhythm", inst_frame, "combobox", 2, col=1)
        self.add_control("min_velocity", "Min Velocity", inst_frame, "slider", 3, slider_range=(1, 127), default=60)
        self.add_control("max_velocity", "Max Velocity", inst_frame, "slider", 3, col=1, slider_range=(1, 127), default=100)

        # --- Generation ---
        gen_frame = ctk.CTkFrame(self.controls_frame)
        gen_frame.pack(expand=True, fill="x", padx=10, pady=10)
        gen_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(gen_frame, text="Generation", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        self.add_control("temperature", "Temperature", gen_frame, "slider", 1, slider_range=(0.1, 1.5), default=0.7, float_steps=True)
        self.add_control("top_p", "Top-P", gen_frame, "slider", 1, col=1, slider_range=(0.1, 1.0), default=0.9, float_steps=True)
        
    def add_control(self, key, label, frame, widget_type, row, col=0, columnspan=1, placeholder="", slider_range=None, default=None, float_steps=False):
        """Helper to create a labeled control widget."""
        label_widget = ctk.CTkLabel(frame, text=label)
        label_widget.grid(row=row*2, column=col, columnspan=columnspan, padx=10, pady=(10, 0), sticky="w")
        
        if widget_type == "combobox":
            widget = ctk.CTkComboBox(frame, values=["-"], state="readonly")
            widget.grid(row=row*2+1, column=col, columnspan=columnspan, padx=10, pady=(0, 10), sticky="ew")
        elif widget_type == "entry":
            widget = ctk.CTkEntry(frame, placeholder_text=placeholder)
            widget.grid(row=row*2+1, column=col, columnspan=columnspan, padx=10, pady=(0, 10), sticky="ew")
        elif widget_type == "slider":
            value_label = ctk.CTkLabel(frame, text=str(default))
            value_label.grid(row=row*2, column=col, padx=(100,10), pady=(10,0), sticky="w")
            widget = ctk.CTkSlider(frame, from_=slider_range[0], to=slider_range[1], command=lambda v, l=value_label: l.configure(text=f"{v:.2f}" if float_steps else f"{int(v)}"))
            widget.set(default)
            widget.grid(row=row*2+1, column=col, columnspan=columnspan, padx=10, pady=(0, 10), sticky="ew")
            self.widgets[f"{key}_value_label"] = value_label

        self.widgets[key] = widget

    def create_output_widgets(self):
        """Creates the widgets for the output/control panel."""
        self.output_frame.grid_rowconfigure(3, weight=1)
        
        ctk.CTkLabel(self.output_frame, text="Controls & Output", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        self.generate_button = ctk.CTkButton(self.output_frame, text="Generate Music", height=40, command=self.start_generation_thread)
        self.generate_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.output_frame, text="Ready", text_color="gray")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5, padx=10, sticky="w")
        
        self.output_textbox = ctk.CTkTextbox(self.output_frame, state="disabled", fg_color="transparent", border_width=1)
        self.output_textbox.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

        self.play_button = ctk.CTkButton(self.output_frame, text="Play Last Generation", state="disabled", command=self.toggle_playback)
        self.play_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        
        open_folder_button = ctk.CTkButton(self.output_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
    
    def load_and_populate_metadata(self):
        """Loads metadata from indexed_tokens_dict.json and populates dropdowns."""
        try:
            with open(self.paths["additional_token_dict_path"], "r") as f:
                tokens = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Metadata Error", f"Could not load or parse token dictionary:\n{e}")
            return

        categories = {}
        for key in tokens:
            if "_" in key:
                # Use rsplit to handle prefixes with underscores like 'track_role'
                parts = key.rsplit('_', 1)
                prefix = parts[0]
                value = parts[1]

                if prefix not in categories:
                    categories[prefix] = []
                categories[prefix].append(value.replace('slash', '/'))
        
        for prefix, values in categories.items():
            # *** FIX: Check if the widget is a ComboBox before trying to configure its values ***
            if prefix in self.widgets and isinstance(self.widgets[prefix], ctk.CTkComboBox):
                try:
                    # Attempt a natural sort for values that might be numbers
                    sorted_values = sorted(values, key=lambda x: int(x.split('/')[0]))
                except ValueError:
                    # Fallback to alphabetical sort if values are not purely numeric
                    sorted_values = sorted(values)
                    
                self.widgets[prefix].configure(values=sorted_values)
                if sorted_values:
                    self.widgets[prefix].set(sorted_values[0])
        
        if 'inst' in self.widgets: self.widgets['inst'].set('acoustic_piano')
        if 'track_role' in self.widgets: self.widgets['track_role'].set('main_melody')

    def start_generation_thread(self):
        """Starts the music generation process in a separate thread to avoid freezing the GUI."""
        self.status_label.configure(text="Status: Generating...", text_color="yellow")
        self.generate_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        
        thread = threading.Thread(target=self.run_generation_subprocess)
        thread.daemon = True
        thread.start()

    def run_generation_subprocess(self):
        """Constructs the command and runs the generation script in a subprocess."""
        params = {}
        for key, widget in self.widgets.items():
            if "_value_label" in key: continue
            
            if isinstance(widget, ctk.CTkComboBox):
                params[key] = widget.get().replace('/', 'slash')
            elif isinstance(widget, ctk.CTkEntry):
                params[key] = widget.get()
            elif isinstance(widget, ctk.CTkSlider):
                params[key] = widget.get()
        
        output_filename = f"generated_{int(time.time() * 1000)}.mid"
        final_output_path = os.path.join(self.paths["output_dir"], output_filename)

        command = [
            'python', self.paths["script_to_run"],
            '--ckpt_dir', self.paths["ckpt_dir"],
            '--tokenizer_path', self.paths["tokenizer_path"],
            '--model_config_path', self.paths["model_config_path"],
            '--additional_token_dict_path', self.paths["additional_token_dict_path"],
            '--chord_dict_path', self.paths["chord_dict_path"],
            '--finetuned_PEFT_weight_path', self.paths["finetuned_PEFT_weight_path"],
            '--output_path', final_output_path,
            '--audio_key', str(params['audio_key']),
            '--pitch_range', str(params['pitch_range']),
            '--num_measures', str(params['num_measures']),
            '--bpm', str(int(params['bpm'])),
            '--genre', str(params['genre']),
            '--track_role', str(params['track_role']),
            '--inst', str(params['inst']),
            '--sample_rhythm', str(params['sample_rhythm']),
            '--time_signature', str(params['time_signature']),
            '--min_velocity', str(int(params['min_velocity'])),
            '--max_velocity', str(int(params['max_velocity'])),
            '--temperature', f"{params['temperature']:.2f}",
            '--top_p', f"{params['top_p']:.2f}",
            '--number_generations', '1',
        ]
        if params.get('chords'):
            command.extend(['--chords', params['chords']])
        
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True,
                cwd=self.paths["project_src_path"], shell=False
            )
            self.after(0, self.on_generation_success, final_output_path, result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_details = e.stderr if hasattr(e, 'stderr') else str(e)
            self.after(0, self.on_generation_failure, error_details)

    def on_generation_success(self, file_path, stdout):
        self.status_label.configure(text="Status: Success!", text_color="lightgreen")
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("end", f"Successfully generated:\n{file_path}\n\n---MODEL OUTPUT---\n{stdout}")
        self.output_textbox.configure(state="disabled")
        self.generate_button.configure(state="normal")
        self.last_generated_file = file_path
        self.play_button.configure(state="normal")

    def on_generation_failure(self, error_details):
        self.status_label.configure(text="Status: Failed!", text_color="red")
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("end", f"--- ERROR ---\n{error_details}")
        self.output_textbox.configure(state="disabled")
        self.generate_button.configure(state="normal")
        messagebox.showerror("Generation Failed", "The script failed to run. Check the output panel for error details.")

    def open_output_folder(self):
        output_dir = self.paths["output_dir"]
        if os.path.exists(output_dir):
            if os.name == 'nt': # Windows
                os.startfile(output_dir)
            elif hasattr(os, 'uname') and os.uname().sysname == 'Darwin': # macOS
                subprocess.run(['open', output_dir])
            else: # Linux
                subprocess.run(['xdg-open', output_dir])
        else:
            messagebox.showwarning("Not Found", "Output directory does not exist yet. Generate a file first.")
    
    def toggle_playback(self):
        if not self.last_generated_file:
            return
        
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.play_button.configure(text="Play Last Generation")
        else:
            try:
                pygame.mixer.music.load(self.last_generated_file)
                pygame.mixer.music.play()
                self.play_button.configure(text="Stop Playback")
                self.check_playback_status()
            except pygame.error as e:
                messagebox.showerror("Playback Error", f"Could not play MIDI file: {e}")

    def check_playback_status(self):
        """Periodically checks if the music has finished playing to reset the button."""
        if pygame.mixer.music.get_busy():
            self.after(1000, self.check_playback_status)
        else:
            self.play_button.configure(text="Play Last Generation")


if __name__ == "__main__":
    app = MidiGeneratorApp()
    app.mainloop()
