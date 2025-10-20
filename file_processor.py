"""
File processing utilities for CSV and Excel files
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import streamlit as st

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles reading different file formats"""

    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        """Read CSV or Excel file and return DataFrame"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == ".csv":
                # Try different encodings and separators
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    for sep in [",", ";", "\t"]:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Successfully parsed
                                logger.info(
                                    f"Successfully read CSV with encoding={encoding}, sep='{sep}'"
                                )
                                return df
                        except:
                            continue
                raise Exception(
                    "Could not read CSV file with any encoding/separator combination"
                )

            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                logger.info(f"Successfully read Excel file: {file_path}")
                return df
            else:
                raise Exception(f"Unsupported file format: {file_ext}")

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def read_uploaded_file(uploaded_file) -> pd.DataFrame:
        """Read uploaded file from Streamlit file uploader"""
        try:
            if uploaded_file.name.endswith(".csv"):
                # Try different encodings and separators for CSV
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    for sep in [",", ";", "\t"]:
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Successfully parsed
                                logger.info(
                                    f"Successfully read CSV with encoding={encoding}, sep='{sep}'"
                                )
                                return df
                        except Exception as e:
                            logger.debug(
                                f"Failed with encoding={encoding}, sep='{sep}': {e}"
                            )
                            continue

                # If all attempts failed, try with default settings
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                return pd.read_excel(uploaded_file)
            else:
                raise Exception(f"Unsupported file format: {uploaded_file.name}")

        except Exception as e:
            logger.error(f"Error reading uploaded file {uploaded_file.name}: {e}")
            raise

    @staticmethod
    def analyze_file_structure(df: pd.DataFrame) -> Dict:
        """Analyze file structure and return summary"""
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "sample_data": {},
            "data_types": {},
            "missing_values": {},
            "numeric_columns": [],
            "text_columns": [],
        }

        # Get sample data (first non-null value for each column)
        for col in df.columns:
            first_valid_index = df[col].first_valid_index()
            if first_valid_index is not None:
                analysis["sample_data"][col] = df.loc[first_valid_index, col]
            else:
                analysis["sample_data"][col] = None

            # Data type analysis
            analysis["data_types"][col] = str(df[col].dtype)
            analysis["missing_values"][col] = df[col].isnull().sum()

            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis["numeric_columns"].append(col)
            else:
                analysis["text_columns"].append(col)

        return analysis

    @staticmethod
    def get_column_preview(df: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
        """Get a preview of the dataframe for display"""
        return df.head(max_rows)

    @staticmethod
    def validate_file_size(uploaded_file, max_size_mb: int = 50) -> bool:
        """Validate uploaded file size"""
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            st.error(
                f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
            )
            return False
        return True

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported file formats"""
        return [".csv", ".xlsx", ".xls"]

    @staticmethod
    def save_uploaded_file(uploaded_file, save_path: str) -> str:
        """Save uploaded file to local path"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info(f"Saved uploaded file to: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise

    @staticmethod
    def detect_delimiter(file_content: str) -> str:
        """Detect the most likely delimiter in CSV content"""
        delimiters = [",", ";", "\t", "|"]
        delimiter_counts = {}

        # Count occurrences of each delimiter in first few lines
        lines = file_content.split("\n")[:5]
        for delimiter in delimiters:
            count = sum(line.count(delimiter) for line in lines)
            delimiter_counts[delimiter] = count

        # Return delimiter with highest count
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        return best_delimiter if delimiter_counts[best_delimiter] > 0 else ","

    @staticmethod
    def get_file_info(uploaded_file) -> Dict:
        """Get information about uploaded file"""
        return {
            "filename": uploaded_file.name,
            "size_mb": uploaded_file.size / (1024 * 1024),
            "type": uploaded_file.type,
            "extension": Path(uploaded_file.name).suffix.lower(),
        }
