"""
Histogram widget component for medical image enhancement application.
Displays image histograms with customization options.
"""
import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QCheckBox, QFrame, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QBrush
from PyQt6.QtCore import Qt, QSize, QRect, QPoint, pyqtSignal, QPointF
# Added QPainterPath import to fix missing QPainterPath reference
from PyQt6.QtGui import QPainterPath

logger = logging.getLogger(__name__)

class HistogramWidget(QWidget):
    """
    Widget for displaying image histograms.
    
    Features:
    - Display grayscale or color histograms
    - Customizable appearance
    - Log or linear scale options
    - Channel selection for color images
    """
    
    def __init__(self, parent=None, channel="gray", bins=256, log_scale=False):
        """
        Initialize the histogram widget.
        
        Args:
            parent: Parent widget
            channel: Channel to display ("gray", "red", "green", "blue", "rgb")
            bins: Number of histogram bins
            log_scale: Whether to use log scale for y-axis
        """
        super().__init__(parent)
        self.channel = channel
        self.bins = bins
        self.log_scale = log_scale
        self.histogram_data = None
        self.histogram_range = (0, 1)
        self.title = "Histogram"
        self.max_value = 0
        self.image_data = None
        
        # Set minimum size
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Initialize appearance settings
        self._init_appearance()
    
    def _init_appearance(self):
        """Initialize appearance settings."""
        # Colors for different channels
        self.channel_colors = {
            "gray": QColor(80, 80, 80),
            "red": QColor(220, 50, 50),
            "green": QColor(50, 180, 50),
            "blue": QColor(50, 80, 220),
            "rgb": QColor(80, 80, 80)  # Used for overlay label
        }
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(245, 245, 245))
        self.setPalette(palette)
        
        # Border settings
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setLineWidth(1)
    
    def set_image(self, image_data):
        """
        Set the image data and calculate histogram.
        
        Args:
            image_data: Numpy array of image data
        """
        if image_data is None:
            return
        
        self.image_data = image_data
        
        # Calculate histogram based on image type and selected channel
        if len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1):
            # Grayscale image
            self.histogram_data = self._calculate_histogram(image_data, "gray")
            self.channel = "gray"
        else:
            # Color image
            if self.channel == "rgb":
                # Calculate histograms for all channels
                self.histogram_data = {
                    "red": self._calculate_histogram(image_data[:,:,0], "red"),
                    "green": self._calculate_histogram(image_data[:,:,1], "green"),
                    "blue": self._calculate_histogram(image_data[:,:,2], "blue")
                }
            else:
                # Calculate histogram for selected channel only
                if self.channel == "gray":
                    # Convert to grayscale first (average of channels)
                    if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                        gray_image = np.mean(image_data[:,:,:3], axis=2)
                        self.histogram_data = self._calculate_histogram(gray_image, "gray")
                elif self.channel in ["red", "green", "blue"]:
                    channel_idx = {"red": 0, "green": 1, "blue": 2}[self.channel]
                    if len(image_data.shape) == 3 and image_data.shape[2] > channel_idx:
                        self.histogram_data = self._calculate_histogram(
                            image_data[:,:,channel_idx], self.channel
                        )
        
        # Update display
        self.update()
    
    def _calculate_histogram(self, channel_data, channel_name):
        """
        Calculate histogram for a single channel.
        
        Args:
            channel_data: Single channel image data
            channel_name: Name of the channel ("gray", "red", "green", "blue")
            
        Returns:
            dict: Histogram data
        """
        # Determine range based on data type
        if np.issubdtype(channel_data.dtype, np.floating):
            hist_range = (0.0, 1.0)
        else:
            hist_range = (0, 255)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(
            channel_data.flatten(), 
            bins=self.bins, 
            range=hist_range
        )
        
        # Store max value for scaling
        max_value = np.max(hist)
        if max_value > self.max_value:
            self.max_value = max_value
        
        # Apply log scale if needed
        if self.log_scale:
            hist = np.log1p(hist)  # log(1+x) to handle zeros
            max_value = np.max(hist)
        
        return {
            "counts": hist,
            "bin_edges": bin_edges,
            "max_value": max_value,
            "channel": channel_name
        }
    
    def set_channel(self, channel):
        """
        Set the channel to display.
        
        Args:
            channel: Channel name ("gray", "red", "green", "blue", "rgb")
        """
        if channel != self.channel:
            self.channel = channel
            
            # Recalculate histogram if needed
            if self.image_data is not None:
                self.set_image(self.image_data)
            else:
                self.update()
    
    def set_log_scale(self, use_log_scale):
        """
        Set whether to use log scale for y-axis.
        
        Args:
            use_log_scale: Whether to use log scale
        """
        if use_log_scale != self.log_scale:
            self.log_scale = use_log_scale
            
            # Recalculate histogram if needed
            if self.image_data is not None:
                self.set_image(self.image_data)
            else:
                self.update()
    
    def set_bins(self, bins):
        """
        Set the number of histogram bins.
        
        Args:
            bins: Number of bins
        """
        if bins != self.bins:
            self.bins = bins
            
            # Recalculate histogram if needed
            if self.image_data is not None:
                self.set_image(self.image_data)
            else:
                self.update()
    
    def set_title(self, title):
        """
        Set the histogram title.
        
        Args:
            title: Title string
        """
        self.title = title
        self.update()
    
    def paintEvent(self, event):
        """Paint the histogram."""
        if self.histogram_data is None:
            # Draw empty histogram
            painter = QPainter(self)
            painter.fillRect(event.rect(), QColor(245, 245, 245))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(
                event.rect(), 
                Qt.AlignmentFlag.AlignCenter,
                "No data"
            )
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(event.rect(), QColor(252, 252, 252))
        
        # Calculate drawing area
        padding = 25
        title_height = 20
        rect = event.rect().adjusted(padding, title_height + 5, -padding, -padding)
        
        # Draw border around histogram area
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawRect(rect)
        
        # Draw title
        painter.setPen(QColor(80, 80, 80))
        title_font = QFont()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRect(rect.left(), 5, rect.width(), title_height),
            Qt.AlignmentFlag.AlignCenter,
            self.title
        )
        
        # Draw the histogram(s)
        if self.channel == "rgb" and isinstance(self.histogram_data, dict):
            # Draw all RGB channels
            for channel, hist in [
                ("red", self.histogram_data.get("red")),
                ("green", self.histogram_data.get("green")),
                ("blue", self.histogram_data.get("blue"))
            ]:
                if hist is not None:
                    self._draw_histogram_channel(painter, rect, hist, channel)
                    
            # Draw legend
            legend_x = rect.right() - 80
            legend_y = rect.top() + 20
            
            for i, channel in enumerate(["red", "green", "blue"]):
                painter.setPen(self.channel_colors[channel])
                painter.setBrush(self.channel_colors[channel])
                
                # Draw color box
                painter.drawRect(legend_x, legend_y + i*20, 10, 10)
                
                # Draw channel name
                painter.drawText(legend_x + 15, legend_y + i*20 + 10, channel.capitalize())
        else:
            # Draw single channel
            if isinstance(self.histogram_data, dict):
                # Single histogram
                self._draw_histogram_channel(painter, rect, self.histogram_data, self.channel)
            else:
                # No histogram data
                painter.setPen(QColor(120, 120, 120))
                painter.drawText(
                    rect, 
                    Qt.AlignmentFlag.AlignCenter,
                    "No histogram data"
                )
        
        # Draw axes labels (value range)
        painter.setPen(QColor(120, 120, 120))
        small_font = QFont()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        
        # X-axis labels (min/max values)
        bin_edges = self.histogram_data["bin_edges"] if isinstance(self.histogram_data, dict) else None
        if bin_edges is not None:
            min_val = bin_edges[0]
            max_val = bin_edges[-1]
            
            if np.issubdtype(type(min_val), np.floating):
                # Format as float with 2 decimal places
                min_label = f"{min_val:.2f}"
                max_label = f"{max_val:.2f}"
            else:
                # Format as integer
                min_label = str(int(min_val))
                max_label = str(int(max_val))
            
            painter.drawText(
                rect.left() - 5,
                rect.bottom() + 15,
                min_label
            )
            
            painter.drawText(
                rect.right() - 20,
                rect.bottom() + 15,
                max_label
            )
        
        # Y-axis label (count)
        if self.log_scale:
            y_label = "Log Count"
        else:
            y_label = "Count"
            
        # Vertical text for y-axis
        painter.save()
        painter.translate(rect.left() - 20, rect.top() + rect.height() // 2)
        painter.rotate(-90)
        painter.drawText(
            QRect(-50, -10, 100, 20),
            Qt.AlignmentFlag.AlignCenter,
            y_label
        )
        painter.restore()
    
    def _draw_histogram_channel(self, painter, rect, hist_data, channel):
        """
        Draw a single histogram channel.
        
        Args:
            painter: QPainter instance
            rect: QRect for drawing area
            hist_data: Histogram data dictionary
            channel: Channel name
        """
        if not hist_data or "counts" not in hist_data:
            return
            
        counts = hist_data["counts"]
        max_value = hist_data["max_value"]
        
        if max_value == 0:
            return
            
        # Set pen color based on channel
        color = self.channel_colors.get(channel, QColor(0, 0, 0))
        pen = QPen(color)
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Calculate bar width
        bar_width = rect.width() / len(counts)
        
        # Draw the histogram
        if bar_width <= 2:
            # Draw as a line graph for narrow histograms
            points = []
            for i, count in enumerate(counts):
                x = rect.left() + i * bar_width
                y = rect.bottom() - (count / max_value) * rect.height()
                points.append(QPoint(int(x), int(y)))
            
            # Draw line connecting points
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
                
            # Fill under the curve with semi-transparent color
            path = QPainterPath()
            path.addPolygon([QPoint(rect.left(), rect.bottom())] + points + [QPoint(rect.right(), rect.bottom())])
            
            # Set semi-transparent fill color
            fill_color = QColor(color)
            fill_color.setAlpha(80)  # 30% opacity
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.PenStyle.NoPen)  # No border when filling
            painter.drawPath(path)
            
            # Restore pen for next drawing
            painter.setPen(pen)
        else:
            # Draw as bars
            for i, count in enumerate(counts):
                bar_height = (count / max_value) * rect.height()
                
                # Only draw visible bars (optimization)
                if bar_height > 0.5:  # Minimum visible height
                    x = rect.left() + i * bar_width
                    y = rect.bottom() - bar_height
                    
                    # Fill bar with semi-transparent color
                    fill_color = QColor(color)
                    fill_color.setAlpha(150)  # 60% opacity
                    painter.fillRect(
                        QRect(int(x), int(y), max(1, int(bar_width - 1)), int(bar_height)),
                        fill_color
                    )
                    
                    # Draw bar outline
                    if bar_width > 3:  # Only draw outline for wider bars
                        painter.setPen(color)
                        painter.drawRect(
                            QRect(int(x), int(y), max(1, int(bar_width - 1)), int(bar_height))
                        )
                        
    def sizeHint(self):
        """Suggested size for this widget."""
        return QSize(300, 200)