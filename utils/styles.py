"""
Style definitions for the medical image enhancement application.
Provides consistent styling across the application.
"""
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtCore import Qt

# Color schemes
class ColorScheme:
    """Color schemes for the application."""
    
    # Medical theme - clean, professional look with medical blue tones
    MEDICAL = {
        "primary": "#0078D7",      # Medium blue - primary UI elements
        "secondary": "#00B0F0",    # Light blue - secondary UI elements
        "accent": "#107C10",       # Green - success/confirmation elements
        "warning": "#FFB900",      # Amber - warning elements
        "error": "#E81123",        # Red - error elements
        "background": "#FFFFFF",   # White - main background
        "card_background": "#F5F5F5", # Light gray - card/panel background
        "text_primary": "#333333", # Dark gray - primary text
        "text_secondary": "#666666", # Medium gray - secondary text
        "border": "#DDDDDD"        # Light gray - borders
    }
    
    # Dark theme - for users who prefer darker interfaces
    DARK = {
        "primary": "#0078D7",
        "secondary": "#00B0F0",
        "accent": "#107C10",
        "warning": "#FFB900",
        "error": "#E81123",
        "background": "#1E1E1E",
        "card_background": "#252525",
        "text_primary": "#FFFFFF",
        "text_secondary": "#BBBBBB",
        "border": "#444444"
    }
    
    # High contrast theme for accessibility
    HIGH_CONTRAST = {
        "primary": "#1AEBFF",
        "secondary": "#3FF23F",
        "accent": "#FFFF00",
        "warning": "#FFB900",
        "error": "#FF0000",
        "background": "#000000",
        "card_background": "#000000",
        "text_primary": "#FFFFFF",
        "text_secondary": "#FFFFFF",
        "border": "#FFFFFF"
    }

def apply_stylesheet(app, theme="medical"):
    """
    Apply stylesheet to the application.
    
    Args:
        app: QApplication instance
        theme: Theme name ('medical', 'dark', or 'high_contrast')
    """
    # Select color scheme
    if theme.lower() == "dark":
        colors = ColorScheme.DARK
    elif theme.lower() == "high_contrast":
        colors = ColorScheme.HIGH_CONTRAST
    else:
        colors = ColorScheme.MEDICAL  # Default to medical theme
    
    # Create stylesheet
    stylesheet = f"""
    /* Global styles */
    QWidget {{
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 10pt;
    }}
    
    /* Main window */
    QMainWindow {{
        background-color: {colors["background"]};
    }}
    
    /* Menu bar */
    QMenuBar {{
        background-color: {colors["background"]};
        color: {colors["text_primary"]};
        border-bottom: 1px solid {colors["border"]};
    }}
    
    QMenuBar::item:selected {{
        background-color: {colors["primary"]};
        color: white;
    }}
    
    /* Menu */
    QMenu {{
        background-color: {colors["background"]};
        color: {colors["text_primary"]};
        border: 1px solid {colors["border"]};
    }}
    
    QMenu::item:selected {{
        background-color: {colors["primary"]};
        color: white;
    }}
    
    /* Toolbar */
    QToolBar {{
        background-color: {colors["background"]};
        border-bottom: 1px solid {colors["border"]};
        spacing: 3px;
    }}
    
    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 3px;
    }}
    
    QToolButton:hover {{
        background-color: rgba(0, 120, 215, 0.1);
        border: 1px solid {colors["primary"]};
    }}
    
    QToolButton:pressed {{
        background-color: rgba(0, 120, 215, 0.2);
    }}
    
    QToolButton:checked {{
        background-color: rgba(0, 120, 215, 0.2);
        border: 1px solid {colors["primary"]};
    }}
    
    /* Group box */
    QGroupBox {{
        background-color: {colors["card_background"]};
        border: 1px solid {colors["border"]};
        border-radius: 4px;
        margin-top: 16px;
        font-weight: bold;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 3px;
        color: {colors["primary"]};
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {colors["primary"]};
        color: white;
        border: none;
        border-radius: 3px;
        padding: 6px 12px;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: #0063B1;  /* Darker blue */
    }}
    
    QPushButton:pressed {{
        background-color: #004E8C;  /* Even darker blue */
    }}
    
    QPushButton:disabled {{
        background-color: #CCCCCC;
        color: #888888;
    }}
    
    /* Secondary button style */
    QPushButton[secondary="true"] {{
        background-color: {colors["card_background"]};
        color: {colors["primary"]};
        border: 1px solid {colors["primary"]};
    }}
    
    QPushButton[secondary="true"]:hover {{
        background-color: rgba(0, 120, 215, 0.1);
    }}
    
    /* Combo box */
    QComboBox {{
        background-color: white;
        border: 1px solid {colors["border"]};
        border-radius: 3px;
        padding: 4px;
        min-width: 6em;
    }}
    
    QComboBox:hover {{
        border: 1px solid {colors["primary"]};
    }}
    
    QComboBox:focus {{
        border: 1px solid {colors["primary"]};
    }}
    
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left: none;
    }}
    
    /* Checkbox */
    QCheckBox {{
        color: {colors["text_primary"]};
        spacing: 5px;
    }}
    
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {colors["border"]};
        border-radius: 3px;
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {colors["primary"]};
        border-color: {colors["primary"]};
    }}
    
    QCheckBox::indicator:hover {{
        border-color: {colors["primary"]};
    }}
    
    /* Slider */
    QSlider::groove:horizontal {{
        border: 1px solid {colors["border"]};
        height: 4px;
        background: {colors["card_background"]};
        margin: 0px;
        border-radius: 2px;
    }}
    
    QSlider::handle:horizontal {{
        background: {colors["primary"]};
        border: none;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: #0063B1;  /* Darker blue */
    }}
    
    /* Status bar */
    QStatusBar {{
        background-color: {colors["background"]};
        color: {colors["text_primary"]};
        border-top: 1px solid {colors["border"]};
    }}
    
    /* Graphics view (for image display) */
    QGraphicsView {{
        background-color: {colors["card_background"]};
        border: 1px solid {colors["border"]};
    }}
    
    /* Scroll bars */
    QScrollBar:vertical {{
        background: {colors["card_background"]};
        width: 12px;
        margin: 12px 0 12px 0;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: #BBBBBB;
        min-height: 20px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: #999999;
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        background: none;
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background: {colors["card_background"]};
        height: 12px;
        margin: 0 12px 0 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: #BBBBBB;
        min-width: 20px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: #999999;
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        background: none;
        width: 0px;
    }}
    
    /* Progress bar */
    QProgressBar {{
        border: 1px solid {colors["border"]};
        border-radius: 3px;
        text-align: center;
        background-color: {colors["card_background"]};
    }}
    
    QProgressBar::chunk {{
        background-color: {colors["primary"]};
        width: 1px;
    }}
    
    /* Splitter */
    QSplitter::handle {{
        background-color: {colors["border"]};
    }}
    
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    """
    
    app.setStyleSheet(stylesheet)
    
    # Set palette for elements that don't use stylesheets
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["background"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["card_background"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["background"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    
    app.setPalette(palette)

def get_medical_logo():
    """
    Get the SVG content for a medical-themed logo.
    Can be used for splash screen, about dialog, etc.
    
    Returns:
        str: SVG content
    """
    # Simple medical-themed logo SVG
    svg = '''
    <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="200" height="200" fill="#ffffff" rx="20" ry="20"/>
      <circle cx="100" cy="100" r="80" fill="#0078D7" opacity="0.1"/>
      <path d="M150,100 A50,50 0 1,1 50,100 A50,50 0 1,1 150,100 Z" fill="none" stroke="#0078D7" stroke-width="6"/>
      <line x1="100" y1="70" x2="100" y2="130" stroke="#0078D7" stroke-width="10" stroke-linecap="round"/>
      <line x1="70" y1="100" x2="130" y2="100" stroke="#0078D7" stroke-width="10" stroke-linecap="round"/>
      <text x="100" y="180" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#0078D7">Medical Image Enhancement</text>
    </svg>
    '''
    return svg

def get_icon_color(theme="medical"):
    """
    Get the primary color for icons based on theme.
    
    Args:
        theme: Theme name ('medical', 'dark', or 'high_contrast')
        
    Returns:
        str: Hex color code
    """
    if theme.lower() == "dark":
        return ColorScheme.DARK["primary"]
    elif theme.lower() == "high_contrast":
        return ColorScheme.HIGH_CONTRAST["primary"]
    else:
        return ColorScheme.MEDICAL["primary"]  # Default to medical theme