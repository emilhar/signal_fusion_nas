import os
from datetime import datetime

class Logger:
    _log_directory = "_misc/long_reverse_experiments"
    _current_log_file = None

    @classmethod
    def _ensure_log_file_exists(cls):
        if cls._current_log_file is None:
            os.makedirs(cls._log_directory, exist_ok=True)
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cls._current_log_file = f"{cls._log_directory}/{current_time}.md"
            
            with open(cls._current_log_file, 'w') as f:
                f.write(f"# New Experiment Started at {current_time}\n\n")

    @classmethod
    def log_new_experiment_heading(cls):
        """Create a new directory if needed under the "_misc" folder named "long reverse experiments". 
        Put a markdown file named the current datetime.now() in it with a header."""
        cls._ensure_log_file_exists()

    @classmethod
    def log_ensemble(cls, target_ranking: list[tuple[str, str]], fake: bool):
        """Place a line in the file with the current datetime and target_ranking, say if it's fake or not"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fake_status = "FAKE" if fake else "REAL"
            ranking_str = ", ".join([f"{t[0]}:{t[1]}" for t in target_ranking])
            f.write(f"- [{timestamp}] Ensemble ranking ({fake_status}): {ranking_str}\n")

    @classmethod
    def log_ea_start(cls, target: str):
        """Place a line in the file with the current datetime and stating that a new evolutionary algorithm has started for the target"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- [{timestamp}] Starting EA for target: {target}\n")

    @classmethod
    def log_ea_completion(cls, stats):
        """Place a line in the file with the current datetime and stating that the evolutionary algorithm has completed with some stats"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- [{timestamp}] EA completed. Stats: {stats}\n")

    @classmethod
    def log_successful_upgrade(cls):
        """Current datetime and the fact that we successfully moved things from temp_models to saved_models"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- [{timestamp}] Successful upgrade: moved from temp_models to saved_models\n")

    @classmethod
    def log_failed_upgrade(cls, old_filter_count: int, new_filter_count: int):
        """Say that we failed to upgrade, so we're moving from the old filter count to the new one"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- [{timestamp}] Upgrade failed. Changing filter count from {old_filter_count} to {new_filter_count}\n")

    @classmethod
    def log_completion(cls, target_ranking: list[tuple[str, str]]):
        """Current date and time, say that the experiment has completed with the target_ranking"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ranking_str = ", ".join([f"{t[0]}:{t[1]}" for t in target_ranking])
            f.write(f"\n## [{timestamp}] Experiment completed with final ranking: {ranking_str}\n")

    @classmethod
    def log_line(cls, line:str):
        """Say that we failed to upgrade, so we're moving from the old filter count to the new one"""
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"- [{timestamp}] {line}\n")

    @classmethod
    def log_ea_logbook(cls, logbook, signal, target):
        cls._ensure_log_file_exists()
        with open(cls._current_log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n- [{timestamp}]({signal}, {target}) EA Generations Summary:\n")
            f.write(f"{" | ".join(logbook[0].keys())}\n")
            f.write("-" * 70 + "\n")
            for gen_data in logbook:
                line = []
                for key in gen_data.keys():
                    line.append(str(gen_data[key]))

                
                f.write(f"{" ".join(line)}")
                f.write("\n")