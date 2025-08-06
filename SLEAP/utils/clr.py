class Clr:
    def __init__(self, string, color=None, bg_color=None):
        self.string = str(string)
        self.color = color.lower() if color else None
        self.bg_color = bg_color.lower() if bg_color else None
    
    def __str__(self):
        fg_colors = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_black": "\033[90m",
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
        }
        
        bg_colors = {
            "black": "\033[40m",
            "red": "\033[41m",
            "green": "\033[42m",
            "yellow": "\033[43m",
            "blue": "\033[44m",
            "magenta": "\033[45m",
            "cyan": "\033[46m",
            "white": "\033[47m",
            "bright_black": "\033[100m",
            "bright_red": "\033[101m",
            "bright_green": "\033[102m",
            "bright_yellow": "\033[103m",
            "bright_blue": "\033[104m",
            "bright_magenta": "\033[105m",
            "bright_cyan": "\033[106m",
            "bright_white": "\033[107m",
        }
        
        codes = []
        
        if self.color and self.color in fg_colors:
            codes.append(fg_colors[self.color])
        
        if self.bg_color and self.bg_color in bg_colors:
            codes.append(bg_colors[self.bg_color])
            
        reset_code = "\033[0m"
        color_code = "".join(codes)
        
        return f"{color_code}{self.string}{reset_code}"
