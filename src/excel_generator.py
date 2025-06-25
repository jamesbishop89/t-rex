"""
Excel Generator Module

Generates compre        # Fonts
        self.fonts = {
            'header': Font(name='Calibri', size=11, bold=True, color='FFFFFF'),
            'difference': Font(name='Calibri', size=10, bold=True, color='CC0000'),
            'normal': Font(name='Calibri', size=10),
            'title': Font(name='Calibri', size=12, bold=True),
            'summary_header': Font(name='Calibri', size=10, bold=True),
            'match_rate_perfect': Font(name='Calibri', size=10, bold=True, color='006400'),  # Dark green
            'match_rate_imperfect': Font(name='Calibri', size=10, bold=True, color='DC143C')  # Dark red
        }Excel output with multiple sheets for reconciliation results.
Includes formatting and detailed analysis.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import io

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Fill, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils.dataframe import dataframe_to_rows

from src.logger_setup import LoggerMixin


class ExcelGenerator(LoggerMixin):
    """
    Generator for comprehensive Excel reconciliation reports.
    
    Creates Excel files with multiple formatted sheets:
    - Summary: Overview with reconciliation and field statistics
    - Matched: Records that match within tolerances
    - Different: Records with differences highlighted
    - Missing in Target: Records only in source
    - Missing in Source: Records only in target
    """
    
    def __init__(self):
        """Initialize Excel generator with formatting styles."""
        self._setup_styles()
    
    def _setup_styles(self) -> None:
        """Set up Excel formatting styles for consistent appearance."""
        # Colors
        self.colors = {
            'header_fill': PatternFill(start_color='366092', end_color='366092', fill_type='solid'),
            'difference_fill': PatternFill(start_color='FFE6E6', end_color='FFE6E6', fill_type='solid'),
            'summary_header': PatternFill(start_color='D9E2F3', end_color='D9E2F3', fill_type='solid'),
            'match_rate_perfect': PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid'),  # Light green
            'match_rate_imperfect': PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')  # Light red
        }
        
        # Fonts
        self.fonts = {
            'header': Font(name='Calibri', size=11, bold=True, color='FFFFFF'),
            'difference': Font(name='Calibri', size=10, bold=True, color='CC0000'),
            'normal': Font(name='Calibri', size=10),
            'summary_header': Font(name='Calibri', size=11, bold=True),
            'title': Font(name='Calibri', size=14, bold=True),
            'match_rate_perfect': Font(name='Calibri', size=10, bold=True, color='006400'),  # Dark green
            'match_rate_imperfect': Font(name='Calibri', size=10, bold=True, color='DC143C')  # Dark red
        }
        
        # Alignments
        self.alignments = {
            'center': Alignment(horizontal='center', vertical='center'),
            'left': Alignment(horizontal='left', vertical='center'),
            'right': Alignment(horizontal='right', vertical='center')
        }
        
        # Borders
        thin_border = Side(border_style='thin', color='000000')
        self.borders = {
            'all': Border(left=thin_border, right=thin_border, top=thin_border, bottom=thin_border)
        }
    
    def generate_excel(self, results: Dict[str, Any], output_path: str, 
                      metadata: Dict[str, Any]) -> None:
        """
        Generate comprehensive Excel report with all reconciliation results.
        
        Args:
            results: Reconciliation results from ReconciliationEngine
            output_path: Path for output Excel file
            metadata: Execution metadata (file paths, timing, etc.)
            
        Raises:
            Exception: If Excel generation fails
        """
        self.logger.info(f"Generating Excel report: {output_path}")
        
        try:
            # Create workbook
            wb = Workbook()
            wb.remove(wb.active)  # Remove default sheet
            
            # Generate all sheets
            self._create_summary_sheet(wb, results, metadata)
            self._create_matched_sheet(wb, results)
            self._create_different_sheet(wb, results)
            self._create_missing_in_target_sheet(wb, results)
            self._create_missing_in_source_sheet(wb, results)
            
            # Save workbook
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)
            
            self.logger.info(f"Excel report generated successfully: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report: {e}")
            raise
    
    def _create_summary_sheet(self, wb: Workbook, results: Dict[str, Any], 
                            metadata: Dict[str, Any]) -> None:
        """
        Create summary sheet with reconciliation and field statistics.
        
        Args:
            wb: Excel workbook
            results: Reconciliation results
            metadata: Execution metadata
        """
        self.logger.info("Creating Summary sheet")
        
        ws = wb.create_sheet("Summary", 0)
        stats = results['statistics']
        
        # Sheet title
        ws['A1'] = 'T-Rex Reconciliation Summary'
        ws['A1'].font = self.fonts['title']
        ws.merge_cells('A1:D1')
        
        # Metadata section
        metadata_data = [
            ['Source File', metadata.get('source_file', 'N/A')],
            ['Target File', metadata.get('target_file', 'N/A')],
            ['Config File', metadata.get('config_file', 'N/A')],
            ['Recon Date', metadata.get('recon_date', 'N/A')],
            ['Execution Time', f"{metadata.get('execution_time', 0):.2f} seconds"]
        ]
        
        start_row = 3
        for i, (label, value) in enumerate(metadata_data):
            row = start_row + i
            ws[f'A{row}'] = label
            ws[f'B{row}'] = value
            ws[f'A{row}'].font = self.fonts['summary_header']
        
        # Statistics section
        stats_start_row = start_row + len(metadata_data) + 2
        ws[f'A{stats_start_row}'] = 'Reconciliation Statistics'
        ws[f'A{stats_start_row}'].font = self.fonts['title']
        
        stats_data = [
            ['Total Source Records', stats['total_source']],
            ['Total Target Records', stats['total_target']],
            ['Matched Records', stats['matched']],
            ['Different Records', stats['different']],
            ['Missing in Source', stats['missing_in_source']],
            ['Missing in Target', stats['missing_in_target']]
        ]
        
        for i, (label, value) in enumerate(stats_data):
            row = stats_start_row + 2 + i
            ws[f'A{row}'] = label
            ws[f'B{row}'] = value
            ws[f'A{row}'].font = self.fonts['summary_header']
          # Field statistics section if available
        if 'field_statistics' in stats and stats['field_statistics']:
            field_stats_start = stats_start_row + len(stats_data) + 4
            ws[f'A{field_stats_start}'] = 'Field Statistics'
            ws[f'A{field_stats_start}'].font = self.fonts['title']
            
            # Headers for field statistics
            headers = ['Field', 'Matches', 'Differences', 'Total Comparable', 'Match Rate']
            for i, header in enumerate(headers):
                col = chr(ord('A') + i)
                ws[f'{col}{field_stats_start + 2}'] = header
                ws[f'{col}{field_stats_start + 2}'].font = self.fonts['summary_header']
                ws[f'{col}{field_stats_start + 2}'].fill = self.colors['summary_header']
            
            # Sort field statistics by differences count (largest to smallest)
            sorted_field_stats = sorted(
                stats['field_statistics'].items(),
                key=lambda x: x[1]['differences_count'],
                reverse=True
            )
              # Field statistics data
            for i, (field_name, field_stats) in enumerate(sorted_field_stats):
                row = field_stats_start + 3 + i
                ws[f'A{row}'] = field_name
                ws[f'B{row}'] = field_stats['matches_count']
                ws[f'C{row}'] = field_stats['differences_count']
                ws[f'D{row}'] = field_stats['total_comparable']
                # Set match rate as number with percentage format
                match_rate_cell = ws[f'E{row}']
                match_rate_cell.value = field_stats['match_rate']
                match_rate_cell.number_format = '0.00%'
                
                # Apply conditional formatting to match rate
                if field_stats['match_rate'] >= 1.0:  # 100% match rate
                    match_rate_cell.fill = self.colors['match_rate_perfect']
                    match_rate_cell.font = self.fonts['match_rate_perfect']
                else:  # Less than 100% match rate
                    match_rate_cell.fill = self.colors['match_rate_imperfect']
                    match_rate_cell.font = self.fonts['match_rate_imperfect']
        
        # Auto-size columns like other sheets
        self._auto_size_summary_columns(ws)
    
    def _create_matched_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """
        Create sheet for matched records.
        
        Args:
            wb: Excel workbook
            results: Reconciliation results
        """
        self.logger.info("Creating Matched sheet")
        
        ws = wb.create_sheet("Matched")
        matched_df = results['records']['matched']
        
        if matched_df.empty:
            ws['A1'] = 'No matched records found'
            return
        
        # Prepare data for display
        display_df = self._prepare_comparison_dataframe(matched_df, results['config'])
        
        # Write data to worksheet
        self._write_dataframe_to_sheet(ws, display_df, freeze_headers=True, add_filters=True)
        
        # Apply formatting
        self._apply_sheet_formatting(ws, display_df)
    
    def _create_different_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """
        Create sheet for different records with difference highlighting.
        
        Args:
            wb: Excel workbook
            results: Reconciliation results
        """
        self.logger.info("Creating Different sheet")
        
        ws = wb.create_sheet("Different")
        different_df = results['records']['different']
        
        if different_df.empty:
            ws['A1'] = 'No different records found'
            return
        
        # Prepare data for display
        display_df = self._prepare_comparison_dataframe(different_df, results['config'])
        
        # Write data to worksheet
        self._write_dataframe_to_sheet(ws, display_df, freeze_headers=True, add_filters=True)
        
        # Apply formatting and highlight differences
        self._apply_sheet_formatting(ws, display_df)
        self._highlight_differences(ws, different_df, results)
    
    def _create_missing_in_target_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """
        Create sheet for records missing in target.
        
        Args:
            wb: Excel workbook
            results: Reconciliation results
        """
        self.logger.info("Creating Missing in Target sheet")
        
        ws = wb.create_sheet("Missing in Target")
        missing_df = results['records']['missing_in_target']
        
        if missing_df.empty:
            ws['A1'] = 'No records missing in target'
            return
        
        # Show only source columns (remove target_ prefix columns)
        source_columns = [col for col in missing_df.columns if col.startswith('source_')]
        display_df = missing_df[source_columns].copy()
        display_df.columns = [col.replace('source_', '') for col in display_df.columns]
        
        # Write data to worksheet
        self._write_dataframe_to_sheet(ws, display_df, freeze_headers=True, add_filters=True)
        
        # Apply formatting
        self._apply_sheet_formatting(ws, display_df)
    
    def _create_missing_in_source_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """
        Create sheet for records missing in source.
        
        Args:
            wb: Excel workbook
            results: Reconciliation results
        """
        self.logger.info("Creating Missing in Source sheet")
        
        ws = wb.create_sheet("Missing in Source")
        missing_df = results['records']['missing_in_source']
        
        if missing_df.empty:
            ws['A1'] = 'No records missing in source'
            return
        
        # Show only target columns (remove target_ prefix columns)
        target_columns = [col for col in missing_df.columns if col.startswith('target_')]
        display_df = missing_df[target_columns].copy()
        display_df.columns = [col.replace('target_', '') for col in display_df.columns]
        
        # Write data to worksheet
        self._write_dataframe_to_sheet(ws, display_df, freeze_headers=True, add_filters=True)        
        # Apply formatting
        self._apply_sheet_formatting(ws, display_df)
    
    def _prepare_comparison_dataframe(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare dataframe for comparison display with key columns first.
        
        Args:
            df: DataFrame to prepare
            config: Configuration dictionary
            
        Returns:
            pd.DataFrame: Prepared dataframe with organized columns
        """
        if df.empty:
            return df
        
        keys = config['reconciliation']['keys']
        fields = config['reconciliation']['fields']
        
        # Start with key columns
        display_columns = keys.copy()
        
        # Add source/target pairs for each configured field (both comparison and ignored)
        for field in fields:
            field_name = field['name']
            source_col = f'source_{field_name}'
            target_col = f'target_{field_name}'
            
            if source_col in df.columns and target_col in df.columns:
                display_columns.extend([source_col, target_col])
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in df.columns]
        display_df = df[available_columns].copy()
          # Rename columns for better display, marking ignored fields
        column_renames = {}
        for col in display_df.columns:
            if col.startswith('source_'):
                field_name = col.replace('source_', '')
                # Check if this field is ignored
                field_config = next((f for f in fields if f['name'] == field_name), None)
                if field_config and field_config.get('ignore', False):
                    column_renames[col] = f"Source: {field_name} (ignored)"
                else:
                    column_renames[col] = f"Source: {field_name}"
            elif col.startswith('target_'):
                field_name = col.replace('target_', '')
                # Check if this field is ignored
                field_config = next((f for f in fields if f['name'] == field_name), None)
                if field_config and field_config.get('ignore', False):
                    column_renames[col] = f"Target: {field_name} (ignored)"
                else:
                    column_renames[col] = f"Target: {field_name}"
        
        display_df = display_df.rename(columns=column_renames)
        
        return display_df
    
    def _write_dataframe_to_sheet(self, ws, df: pd.DataFrame, freeze_headers: bool = True,
                                add_filters: bool = True, auto_size: bool = True) -> None:
        """
        Write DataFrame to Excel worksheet with formatting options.
        
        Args:
            ws: Excel worksheet
            df: DataFrame to write
            freeze_headers: Whether to freeze header row
            add_filters: Whether to add auto filters
            auto_size: Whether to auto-size columns based on content
        """
        # Write data to worksheet
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        
        # Freeze headers if requested
        if freeze_headers and len(df) > 0:
            ws.freeze_panes = 'A2'
              # Add filters if requested
        if add_filters and len(df) > 0 and len(df.columns) > 0:
            # Create proper Excel column reference for any number of columns
            last_col = self._get_excel_column_name(len(df.columns))
            last_row = len(df) + 1
            filter_range = f"A1:{last_col}{last_row}"
            ws.auto_filter.ref = filter_range
            
        # Auto-size columns if requested
        if auto_size:
            self._auto_size_columns(ws, df)
    
    def _apply_sheet_formatting(self, ws, df: pd.DataFrame) -> None:
        """
        Apply consistent formatting to worksheet.
        
        Args:
            ws: Excel worksheet
            df: DataFrame (for determining dimensions)
        """
        if df.empty:
            return
        
        # Format headers
        for col_num in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = self.fonts['header']
            cell.fill = self.colors['header_fill']
            cell.alignment = self.alignments['center']
            cell.border = self.borders['all']
        
        # Format data cells
        for row_num in range(2, len(df) + 2):
            for col_num in range(1, len(df.columns) + 1):
                cell = ws.cell(row=row_num, column=col_num)
                cell.font = self.fonts['normal']
                cell.border = self.borders['all']        # Auto-size columns based on content
        self._auto_size_columns(ws, df)
    
    def _highlight_differences(self, ws, different_df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Highlight cells with differences in the Different sheet.
        
        Args:
            ws: Excel worksheet
            different_df: DataFrame with different records
            results: Reconciliation results including field comparisons
        """
        if different_df.empty:
            return
        
        try:
            config = results['config']
            field_comparison = results.get('field_comparison', {})
            
            # Get column mapping
            display_df = self._prepare_comparison_dataframe(different_df, config)
            
            for field_name, comparison_result in field_comparison.items():
                source_col_name = f"Source: {field_name}"
                target_col_name = f"Target: {field_name}"
                
                if source_col_name not in display_df.columns or target_col_name not in display_df.columns:
                    continue
                
                # Find column indices
                source_col_idx = list(display_df.columns).index(source_col_name) + 1
                target_col_idx = list(display_df.columns).index(target_col_name) + 1
                
                # For each row in different_df, check if this field has a difference and highlight it
                for diff_row_idx in range(len(different_df)):
                    excel_row = diff_row_idx + 2  # +2 for header and 0-based index
                    
                    # Get the actual values for this row to determine if they're different
                    source_value = different_df.iloc[diff_row_idx][f'source_{field_name}'] if f'source_{field_name}' in different_df.columns else None
                    target_value = different_df.iloc[diff_row_idx][f'target_{field_name}'] if f'target_{field_name}' in different_df.columns else None
                    
                    # Check if values are different - handle null/blank comparisons properly
                    values_are_different = self._values_are_different(source_value, target_value)
                    
                    if values_are_different:
                        # Highlight source cell
                        source_cell = ws.cell(row=excel_row, column=source_col_idx)
                        source_cell.fill = self.colors['difference_fill']
                        source_cell.font = self.fonts['difference']
                        
                        # Highlight target cell
                        target_cell = ws.cell(row=excel_row, column=target_col_idx)
                        target_cell.fill = self.colors['difference_fill']
                        target_cell.font = self.fonts['difference']
        
        except Exception as e:
            self.logger.warning(f"Could not highlight differences: {e}")
    
    def _values_are_different(self, source_value, target_value) -> bool:
        """
        Check if two values are different, handling null, blank, and empty string comparisons.
        
        Args:
            source_value: Value from source dataset
            target_value: Value from target dataset
            
        Returns:
            bool: True if values are different, False if they are considered the same
        """
        import pandas as pd
        
        # Handle pandas NaN values
        source_is_na = pd.isna(source_value)
        target_is_na = pd.isna(target_value)
        
        # Both are NaN/None - considered same
        if source_is_na and target_is_na:
            return False
        
        # One is NaN/None, other is not - different
        if source_is_na != target_is_na:
            return True
        
        # Convert to string for comparison (handles various data types)
        source_str = str(source_value).strip() if source_value is not None else ""
        target_str = str(target_value).strip() if target_value is not None else ""
        
        # Compare string representations
        return source_str != target_str
    
    def _get_excel_column_name(self, col_num: int) -> str:
        """
        Convert column number to Excel column name (A, B, ..., Z, AA, AB, etc.).
        
        Args:
            col_num: Column number (1-based)
            
        Returns:
            Excel column name
        """
        result = ""
        while col_num > 0:
            col_num -= 1  # Convert to 0-based
            result = chr(col_num % 26 + ord('A')) + result
            col_num //= 26
        return result
    
    def _auto_size_columns(self, ws, df: pd.DataFrame, min_width: int = 8, max_width: int = 50) -> None:
        """
        Auto-size columns based on content length for better visibility.
        
        Args:
            ws: Excel worksheet
            df: DataFrame to analyze for column sizing
            min_width: Minimum column width
            max_width: Maximum column width
        """
        if df.empty:
            return
            
        for col_num in range(1, len(df.columns) + 1):
            column_letter = self._get_excel_column_name(col_num)
            
            # Get column name and calculate width needed for header
            header_name = df.columns[col_num - 1]
            header_width = len(str(header_name)) + 2  # Add padding
            
            # Calculate width needed for data in this column
            max_data_width = 0
            if len(df) > 0:
                for value in df.iloc[:, col_num - 1]:
                    if pd.notna(value):
                        value_width = len(str(value))
                        max_data_width = max(max_data_width, value_width)
            
            # Use the larger of header or data width, within limits
            optimal_width = max(header_width, max_data_width + 2)  # Add padding
            final_width = max(min_width, min(optimal_width, max_width))
            
            ws.column_dimensions[column_letter].width = final_width
    
    def _auto_size_summary_columns(self, ws, min_width: int = 8, max_width: int = 50) -> None:
        """
        Auto-size columns in summary sheet based on content length.
        
        Args:
            ws: Excel worksheet
            min_width: Minimum column width
            max_width: Maximum column width
        """
        # Get the maximum column index that contains data
        max_col = 1
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    max_col = max(max_col, cell.column)
        
        # Auto-size each column based on content
        for col_num in range(1, max_col + 1):
            column_letter = self._get_excel_column_name(col_num)
            max_width_needed = 0
            
            # Check all cells in this column to find the longest content
            for row in ws.iter_rows(min_col=col_num, max_col=col_num):
                for cell in row:
                    if cell.value is not None:
                        cell_width = len(str(cell.value)) + 2  # Add padding
                        max_width_needed = max(max_width_needed, cell_width)
            
            # Set column width within the specified limits
            final_width = max(min_width, min(max_width_needed, max_width))
            ws.column_dimensions[column_letter].width = final_width
