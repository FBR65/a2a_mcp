import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Für Text- und Bilddateien
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from PIL import Image as PilImage  # Umbenennung, um Konflikte zu vermeiden

# Für HTML-Dateien
try:
    from weasyprint import HTML
except ImportError:
    HTML = None
    # Warnung, wenn WeasyPrint nicht verfügbar ist. Die HTML-Konvertierung wird dann übersprungen.
    logging.warning(
        "WeasyPrint oder seine Abhängigkeiten sind nicht installiert. HTML-Konvertierung wird nicht funktionieren. Bitte 'pip install WeasyPrint' und die externen Abhängigkeiten installieren."
    )


class PDFConverter:
    """
    Eine Klasse zum Konvertieren verschiedener Dateitypen in PDF.

    Unterstützte Dateitypen:
    - Textdateien: .txt, .md, .py, .csv, .log, .json, .xml
    - Bilddateien: .jpg, .png, .jpeg, .gif, .bmp, .tiff, .webp
    - Office-Dokumente: .docx, .xlsx, .pptx (benötigt LibreOffice)
    - HTML-Dateien: .html, .htm (benötigt WeasyPrint)
    """

    def __init__(self):
        """
        Initialisiert den PDFConverter.
        Konfiguriert die Protokollierung.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _get_libreoffice_command(self) -> str:
        """
        Ermittelt den korrekten LibreOffice-Befehl für das aktuelle Betriebssystem.
        Versucht zuerst einen Standardpfad unter Windows, fällt dann auf 'soffice' zurück.
        """
        if os.name == "nt":  # Windows
            # Standardpfad für LibreOffice auf Windows, kann je nach Installation variieren
            libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
            if Path(libreoffice_path).exists():
                return libreoffice_path
            else:
                self.logger.warning(
                    f"LibreOffice-Executable nicht am Standardpfad gefunden: {libreoffice_path}. "
                    "Versuche, 'soffice' direkt aufzurufen (muss in PATH sein)."
                )
                return "soffice"
        else:  # Linux/macOS
            return "soffice"  # Muss in der PATH-Umgebungsvariablen sein

    def _convert_office_to_pdf(
        self, input_filepath: Path, output_filepath: Path
    ) -> bool:
        """
        Konvertiert eine Microsoft Office-Datei (docx, xlsx, pptx) in PDF mithilfe von LibreOffice.
        LibreOffice muss auf dem System installiert sein und in der PATH-Umgebungsvariablen verfügbar sein.
        """
        self.logger.info(f"Versuche, Office-Datei zu konvertieren: {input_filepath}")

        libreoffice_cmd = self._get_libreoffice_command()

        # Befehl zum Ausführen der Konvertierung
        # --headless: Keine Benutzeroberfläche anzeigen
        # --convert-to pdf: Konvertiert die Datei in PDF
        # --outdir: Legt das Ausgabeverzeichnis fest
        command = [
            libreoffice_cmd,
            "--headless",
            "--convert-to",
            "pdf",
            str(input_filepath),
            "--outdir",
            str(output_filepath.parent),
        ]

        try:
            # Führen Sie den Befehl aus
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, encoding="utf-8"
            )
            self.logger.info(f"LibreOffice stdout: {result.stdout.strip()}")
            if result.stderr:
                self.logger.warning(f"LibreOffice stderr: {result.stderr.strip()}")

            # LibreOffice erstellt die Datei im outdir mit dem gleichen Namen, aber .pdf-Endung
            expected_output_filename = input_filepath.stem + ".pdf"
            actual_output_path = output_filepath.parent / expected_output_filename

            # Überprüfen, ob die Datei erstellt wurde und ggf. umbenennen
            if actual_output_path.exists():
                if actual_output_path != output_filepath:
                    actual_output_path.rename(output_filepath)
                self.logger.info(
                    f"Office-Datei erfolgreich konvertiert zu: {output_filepath}"
                )
                return True
            else:
                self.logger.error(
                    f"LibreOffice hat die PDF-Datei nicht erstellt am erwarteten Pfad: {actual_output_path}"
                )
                return False

        except FileNotFoundError:
            self.logger.error(
                f"LibreOffice-Befehl '{libreoffice_cmd}' nicht gefunden. Bitte stellen Sie sicher, dass LibreOffice installiert ist und in Ihrer PATH-Umgebungsvariablen verfügbar ist, oder geben Sie den vollständigen Pfad an."
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Fehler bei der LibreOffice-Konvertierung: {e}")
            self.logger.error(f"Stdout: {e.stdout.strip()}")
            self.logger.error(f"Stderr: {e.stderr.strip()}")
        except Exception as e:
            self.logger.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return False

    def _convert_text_to_pdf(self, input_filepath: Path, output_filepath: Path) -> bool:
        """Konvertiert eine Textdatei in PDF."""
        self.logger.info(f"Versuche, Textdatei zu konvertieren: {input_filepath}")
        try:
            doc = SimpleDocTemplate(str(output_filepath), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Inhalt der Datei lesen
            with open(input_filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Teilen Sie den Inhalt in Absätze auf, um die Formatierung zu verbessern
            for line in content.splitlines():
                # Ein Leerzeichen hinzufügen, wenn die Zeile leer ist, um eine leere Zeile im PDF zu erzwingen
                story.append(Paragraph(line if line.strip() else " ", styles["Normal"]))
                story.append(
                    Spacer(1, 0.2 * inch)
                )  # Kleiner Abstand zwischen den Zeilen

            doc.build(story)
            self.logger.info(f"Textdatei erfolgreich konvertiert zu: {output_filepath}")
            return True
        except Exception as e:
            self.logger.error(
                f"Fehler beim Konvertieren der Textdatei '{input_filepath}': {e}"
            )
        return False

    def _convert_image_to_pdf(
        self, input_filepath: Path, output_filepath: Path
    ) -> bool:
        """Konvertiert eine Bilddatei in PDF."""
        self.logger.info(f"Versuche, Bilddatei zu konvertieren: {input_filepath}")
        try:
            doc = SimpleDocTemplate(str(output_filepath), pagesize=letter)
            story = []

            img = PilImage.open(input_filepath)
            img_width, img_height = img.size

            # Berechnen Sie die Skalierung, um das Bild an die Seite anzupassen
            page_width, page_height = letter

            # Abzüglich Ränder (standardmäßig 1 Zoll auf jeder Seite in ReportLab)
            usable_width = page_width - 2 * inch
            usable_height = page_height - 2 * inch

            aspect_ratio = img_width / img_height

            # Passt das Bild an die Seite an, falls es zu groß ist, unter Beibehaltung des Seitenverhältnisses
            if img_width > usable_width or img_height > usable_height:
                if (usable_width / img_width) < (usable_height / img_height):
                    width = usable_width
                    height = usable_width / aspect_ratio
                else:
                    height = usable_height
                    width = usable_height * aspect_ratio
            else:
                width, height = (
                    img_width,
                    img_height,
                )  # Verwende Originalgröße, wenn sie auf die Seite passt

            # Fügt das Bild zum Dokument hinzu
            reportlab_image = Image(str(input_filepath), width=width, height=height)
            story.append(reportlab_image)

            doc.build(story)
            self.logger.info(f"Bilddatei erfolgreich konvertiert zu: {output_filepath}")
            return True
        except Exception as e:
            self.logger.error(
                f"Fehler beim Konvertieren der Bilddatei '{input_filepath}': {e}"
            )
        return False

    def _convert_html_to_pdf(self, input_filepath: Path, output_filepath: Path) -> bool:
        """Konvertiert eine HTML-Datei in PDF mithilfe von WeasyPrint."""
        self.logger.info(f"Versuche, HTML-Datei zu konvertieren: {input_filepath}")
        if HTML is None:
            self.logger.error(
                "WeasyPrint ist nicht verfügbar. Kann HTML-Dateien nicht konvertieren."
            )
            return False
        try:
            HTML(filename=str(input_filepath)).write_pdf(str(output_filepath))
            self.logger.info(
                f"HTML-Datei erfolgreich konvertiert zu: {output_filepath}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Fehler beim Konvertieren der HTML-Datei '{input_filepath}': {e}"
            )
        return False

    def convert(
        self,
        input_filepath: str,
        output_directory: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Konvertiert eine gegebene Datei in PDF. Die Methode wählt den Konvertierungsalgorithmus
        basierend auf der Dateiendung.

        Args:
            input_filepath (str): Der vollständige Pfad zur Eingabedatei, die konvertiert werden soll.
            output_directory (str, optional): Das Verzeichnis, in dem die PDF-Datei gespeichert werden soll.
                                             Standardmäßig das Verzeichnis der Eingabedatei.
            output_filename (str, optional): Der Dateiname für die generierte PDF-Datei (ohne Erweiterung).
                                            Standardmäßig der Name der Eingabedatei.

        Returns:
            Optional[Path]: Der Pfad zur generierten PDF-Datei (als `pathlib.Path`-Objekt),
                            wenn die Konvertierung erfolgreich war, andernfalls `None`.
        """
        input_path = Path(input_filepath)

        if not input_path.exists():
            self.logger.error(f"Eingabedatei nicht gefunden: {input_filepath}")
            return None

        # Bestimmen Sie das Ausgabeverzeichnis
        if output_directory:
            output_dir_path = Path(output_directory)
        else:
            output_dir_path = input_path.parent

        # Bestimmen Sie den Ausgabedateinamen
        if output_filename:
            # Sicherstellen, dass keine Endung übergeben wird (falls doch, wird sie entfernt)
            output_base_name = Path(output_filename).stem
        else:
            output_base_name = input_path.stem

        output_path = output_dir_path / f"{output_base_name}.pdf"

        # Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_extension = input_path.suffix.lower()
        success = False

        if file_extension in [".docx", ".xlsx", ".pptx"]:
            success = self._convert_office_to_pdf(input_path, output_path)
        elif file_extension in [".txt", ".md", ".py", ".csv", ".log", ".json", ".xml"]:
            success = self._convert_text_to_pdf(input_path, output_path)
        elif file_extension in [
            ".jpg",
            ".png",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        ]:
            success = self._convert_image_to_pdf(input_path, output_path)
        elif file_extension in [".html", ".htm"]:
            success = self._convert_html_to_pdf(input_path, output_path)
        else:
            self.logger.warning(
                f"Dateityp '{file_extension}' wird nicht direkt unterstützt. "
                "Eine Konvertierung mit Formatierungserhalt ist hier schwierig."
            )
            success = (
                False  # Explizit auf False setzen, wenn der Typ nicht unterstützt wird
            )

        if success:
            return output_path
        else:
            return None


if __name__ == "__main__":
    # --- Beispielverwendungen der PDFConverter-Klasse ---

    # Instanz der Klasse erstellen
    converter = PDFConverter()

    # Erstellen Sie einige Dummy-Dateien für Tests
    output_dir = Path("konvertierte_pdfs")
    output_dir.mkdir(
        exist_ok=True
    )  # Stellen Sie sicher, dass das Ausgabeverzeichnis existiert

    # 1. Dummy-Textdatei
    dummy_text_path = output_dir / "beispiel_text.txt"
    with open(dummy_text_path, "w", encoding="utf-8") as f:
        f.write("Dies ist eine Beispiel-Textdatei.\n")
        f.write("Sie enthält mehrere Zeilen.\n")
        f.write("Absätze und Zeilenumbrüche sollten beibehalten werden.\n\n")
        f.write("Ein weiterer Absatz mit etwas mehr Inhalt.")

    # 2. Dummy-HTML-Datei
    dummy_html_path = output_dir / "beispiel_html.html"
    with open(dummy_html_path, "w", encoding="utf-8") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Beispiel HTML</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                p { line-height: 1.5; }
                .highlight { background-color: yellow; padding: 5px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Willkommen zu diesem HTML-Beispiel</h1>
            <p>Dies ist ein Absatz mit <span class="highlight">hervorgehobenem Text</span>.</p>
            <p>Ein weiterer Absatz, um die Formatierung zu testen.</p>
            <ul>
                <li>Listenpunkt 1</li>
                <li>Listenpunkt 2</li>
            </ul>
            <p>Fussnote: Das ist ein Test für WeasyPrint.</p>
        </body>
        </html>
        """)

    # 3. Dummy-Bild (ein kleines leeres Bild erstellen)
    dummy_image_path = output_dir / "beispiel_bild.png"
    try:
        img_placeholder = PilImage.new("RGB", (600, 400), color="lightblue")
        img_placeholder.save(dummy_image_path)
    except Exception as e:
        converter.logger.warning(
            f"Konnte Dummy-Bild nicht erstellen: {e}. Überspringe den Bildtest."
        )
        dummy_image_path = None  # Setzen, damit der Test übersprungen wird

    print("\n--- Testen der Dateikonvertierungen ---")

    # Konvertieren einer Textdatei
    pdf_path_text = converter.convert(
        str(dummy_text_path),
        output_directory=str(output_dir),
        output_filename="ausgabe_text",
    )
    if pdf_path_text:
        print(f"Textdatei erfolgreich konvertiert: {pdf_path_text}")
    else:
        print("Textdatei-Konvertierung fehlgeschlagen.")

    # Konvertieren einer HTML-Datei
    pdf_path_html = converter.convert(
        str(dummy_html_path),
        output_directory=str(output_dir),
        output_filename="ausgabe_html",
    )
    if pdf_path_html:
        print(f"HTML-Datei erfolgreich konvertiert: {pdf_path_html}")
    else:
        print(
            "HTML-Datei-Konvertierung fehlgeschlagen (prüfen Sie WeasyPrint Installation und Abhängigkeiten)."
        )

    # Konvertieren einer Bilddatei
    if dummy_image_path:
        pdf_path_image = converter.convert(
            str(dummy_image_path),
            output_directory=str(output_dir),
            output_filename="ausgabe_bild",
        )
        if pdf_path_image:
            print(f"Bilddatei erfolgreich konvertiert: {pdf_path_image}")
        else:
            print("Bilddatei-Konvertierung fehlgeschlagen.")
    else:
        print(
            "Bildkonvertierungstest übersprungen, da Dummy-Bild nicht erstellt werden konnte."
        )

    # Beispiel für eine Office-Datei (diese Datei muss tatsächlich existieren)
    # Erstellen Sie manuell eine beispiel_office.docx oder beispiel_office.xlsx Datei
    # im Ordner 'konvertierte_pdfs' oder passen Sie den Pfad an.
    dummy_docx_path = output_dir / "beispiel_office.docx"
    if dummy_docx_path.exists():
        print(
            f"\nVersuche, Office-Datei ({dummy_docx_path}) zu konvertieren. Dies erfordert LibreOffice."
        )
        pdf_path_docx = converter.convert(
            str(dummy_docx_path),
            output_directory=str(output_dir),
            output_filename="ausgabe_docx",
        )
        if pdf_path_docx:
            print(f"Office-Datei erfolgreich konvertiert: {pdf_path_docx}")
        else:
            print(
                "Office-Datei-Konvertierung fehlgeschlagen (prüfen Sie LibreOffice Installation und PATH-Variable)."
            )
    else:
        print(
            f"\nÜberspringe Office-Dateikonvertierungstest: '{dummy_docx_path}' nicht gefunden. Bitte erstellen Sie eine Datei, um dies zu testen."
        )

    # Aufräumen der Dummy-Eingabedateien (Die generierten PDFs bleiben erhalten)
    print("\n--- Bereinigung der Dummy-Eingabedateien ---")
    for f_path in [dummy_text_path, dummy_html_path, dummy_image_path]:
        if f_path and f_path.exists():
            f_path.unlink()
            print(f"Gelöscht: {f_path}")

    print("\nAlle Konvertierungsversuche abgeschlossen.")
    print(
        f"Überprüfen Sie die generierten PDF-Dateien im Verzeichnis: {output_dir.absolute()}"
    )
