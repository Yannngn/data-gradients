from dataclasses import dataclass
from pathlib import Path

import seaborn
from jinja2 import Template

try:
    from xhtml2pdf import pisa

    HAS_PISA = True
except ImportError:
    HAS_PISA = False

import data_gradients
from data_gradients.assets import assets
from data_gradients.config.utils import PathLike


@dataclass
class FeatureSummary:
    name: str
    description: str
    image_path: PathLike | None = None
    notice: str | None = None
    warning: str | None = None


class Section:
    def __init__(self, section_name: str) -> None:
        self.section_name = section_name
        self.features: list[FeatureSummary] = []

    def add_feature(self, feature: FeatureSummary) -> None:
        self.features.append(feature)


class ResultsContainer:
    """
    A container for the results of the analysis.
    dived to sections and features.
    """

    def __init__(self) -> None:
        self.sections: list[Section] = []

    def add_section(self, section: Section) -> None:
        self.sections.append(section)


class PDFWriter:
    """
    This class is responsible for generating the PDF file.
    It uses the pisa library to generate the PDF file.

    The PDF file is generated based on HTML templates (document, section and feature templates).
    """

    def __init__(
        self,
        title: str,
        subtitle: str,
        html_template: str = assets.html.doc_template,
        logo_path: PathLike = assets.image.logo,
        palette: str = "pastel",
    ) -> None:
        """
        :param title: The title of the PDF document.
        :param subtitle: The subtitle of the PDF document.
        :param html_template: The path to the document template.
        :param logo_path: The path to the logo image.
        """
        self.title = title
        self.subtitle = subtitle
        self.template = Template(source=html_template)
        self.logo_path = Path(logo_path)
        palette_list = seaborn.color_palette(palette=palette).as_hex()
        self.train_color = palette_list[0]
        self.val_color = palette_list[1]

    def write(self, results_container: ResultsContainer, output_filename: PathLike) -> None:
        """
        :param results_container: The results container containing the sections and features.
        :param output_filename: The path to the output file.
        """
        if not HAS_PISA:
            raise ImportError(
                "PDF generation requires the 'xhtml2pdf' package. Install it with: pip install data-gradients[pdf] or uv sync --group pdf"
            )

        output_path = Path(output_filename)
        if output_path.suffix != ".pdf":
            raise RuntimeError("filename must end with .pdf")

        doc = self.template.render(
            title=self.title,
            subtitle=self.subtitle,
            results=results_container,
            version=data_gradients.__version__,
            train_color=self.train_color,
            val_color=self.val_color,
            logo=self.logo_path,
            assets=assets,
        )

        with output_path.open("w+b") as result_file:
            pisa.CreatePDF(doc, dest=result_file)
