import sys
import logging

from PyQt6.QtWidgets import QApplication

from ui.app import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)


def main() -> None:
    """Entry point for WaveWay."""
    app = QApplication(sys.argv)
    app.setApplicationName("WaveWay")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()