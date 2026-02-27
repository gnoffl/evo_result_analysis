"""
Unit tests for the io module.

Tests the output formatting utilities used across analysis scripts.
"""

import unittest
from unittest.mock import patch, call
from io import StringIO
import sys
from datetime import datetime

from src.analysis.io import print_section_header, print_subsection, print_status


class TestPrintSectionHeader(unittest.TestCase):
    """Tests for the print_section_header function."""
    
    def setUp(self):
        """Redirect stdout to capture print output."""
        self.held_output = StringIO()
        sys.stdout = self.held_output
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = sys.__stdout__
    
    def test_default_symbol(self):
        """Test section header with default '=' symbol."""
        print_section_header("TEST HEADER")
        output = self.held_output.getvalue()
        
        # Should have 4 lines plus trailing empty string: newline, equals line, title, equals line
        lines = output.split('\n')
        self.assertEqual(len(lines), 5)  # Including empty string after final \n
        self.assertEqual(lines[0], '')  # Initial newline
        self.assertEqual(lines[1], '=' * 80)
        self.assertIn("TEST HEADER", lines[2])
        self.assertEqual(lines[3], '=' * 80)
        # Check that title is centered
        self.assertEqual(lines[2].strip(), "TEST HEADER")
        self.assertEqual(len(lines[2]), 80)
    
    def test_custom_symbol(self):
        """Test section header with custom symbol."""
        print_section_header("CUSTOM", "#")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(lines[1], '#' * 80)
        self.assertEqual(lines[3], '#' * 80)
        self.assertIn("CUSTOM", lines[2])
    
    def test_long_title(self):
        """Test section header with a title longer than 80 characters."""
        long_title = "A" * 100
        print_section_header(long_title)
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertIn(long_title, lines[2])
        self.assertEqual(len(lines[1]), 80)  # Border is still 80 chars
    
    def test_empty_title(self):
        """Test section header with empty title."""
        print_section_header("")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(len(lines[2].strip()), 0)  # Empty centered string
        self.assertEqual(len(lines[2]), 80)  # But still 80 chars (spaces)
    
    def test_multiline_symbol(self):
        """Test section header with multi-character symbol."""
        print_section_header("TEST", "=-")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        # Should repeat the pattern to reach 80 chars
        self.assertTrue(lines[1].startswith("=-"))
        self.assertTrue(len(lines[1]) >= 80)


class TestPrintSubsection(unittest.TestCase):
    """Tests for the print_subsection function."""
    
    def setUp(self):
        """Redirect stdout to capture print output."""
        self.held_output = StringIO()
        sys.stdout = self.held_output
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = sys.__stdout__
    
    def test_basic_subsection(self):
        """Test basic subsection output."""
        print_subsection("Configuration")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(len(lines), 5)  # newline, line, title, line, trailing empty
        self.assertEqual(lines[0], '')  # Initial newline
        self.assertEqual(lines[1], '─' * 80)
        self.assertEqual(lines[2], "  Configuration")
        self.assertEqual(lines[3], '─' * 80)
    
    def test_empty_title(self):
        """Test subsection with empty title."""
        print_subsection("")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(lines[2], "  ")  # Just the indent
    
    def test_long_title(self):
        """Test subsection with very long title."""
        long_title = "A" * 200
        print_subsection(long_title)
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(lines[2], f"  {long_title}")
        # Lines should still be 80 chars
        self.assertEqual(len(lines[1]), 80)
        self.assertEqual(len(lines[3]), 80)
    
    def test_special_characters(self):
        """Test subsection with special characters in title."""
        print_subsection("Step 1: Initialize (100%)")
        output = self.held_output.getvalue()
        
        lines = output.split('\n')
        self.assertEqual(lines[2], "  Step 1: Initialize (100%)")


class TestPrintStatus(unittest.TestCase):
    """Tests for the print_status function."""
    
    def setUp(self):
        """Redirect stdout to capture print output."""
        self.held_output = StringIO()
        sys.stdout = self.held_output
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = sys.__stdout__
    
    @patch('src.analysis.io.datetime')
    def test_info_status(self, mock_datetime):
        """Test status message with INFO status."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Processing data", "INFO")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[14:30:45] ℹ Processing data")
    
    @patch('src.analysis.io.datetime')
    def test_success_status(self, mock_datetime):
        """Test status message with SUCCESS status."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Operation completed", "SUCCESS")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[14:30:45] ✓ Operation completed")
    
    @patch('src.analysis.io.datetime')
    def test_error_status(self, mock_datetime):
        """Test status message with ERROR status."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Operation failed", "ERROR")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[14:30:45] ✗ Operation failed")
    
    @patch('src.analysis.io.datetime')
    def test_warning_status(self, mock_datetime):
        """Test status message with WARNING status."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Potential issue detected", "WARNING")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[14:30:45] ⚠ Potential issue detected")
    
    @patch('src.analysis.io.datetime')
    def test_unknown_status(self, mock_datetime):
        """Test status message with unknown status type."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Custom message", "CUSTOM")
        output = self.held_output.getvalue().strip()
        
        # Should use default bullet symbol
        self.assertEqual(output, "[14:30:45] • Custom message")
    
    @patch('src.analysis.io.datetime')
    def test_default_status(self, mock_datetime):
        """Test status message with default status (no status parameter)."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Default message")
        output = self.held_output.getvalue().strip()
        
        # Should use INFO symbol by default
        self.assertEqual(output, "[14:30:45] ℹ Default message")
    
    @patch('src.analysis.io.datetime')
    def test_empty_message(self, mock_datetime):
        """Test status with empty message."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("", "INFO")
        output = self.held_output.getvalue().strip()
        
        # Empty message still has trailing space after symbol
        self.assertEqual(output, "[14:30:45] ℹ")
    
    @patch('src.analysis.io.datetime')
    def test_multiline_message(self, mock_datetime):
        """Test status with multiline message."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_status("Line 1\nLine 2\nLine 3", "INFO")
        output = self.held_output.getvalue().strip()
        
        # Should print as-is with newlines
        self.assertEqual(output, "[14:30:45] ℹ Line 1\nLine 2\nLine 3")
    
    @patch('src.analysis.io.datetime')
    def test_case_sensitivity(self, mock_datetime):
        """Test that status is case-sensitive."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        # Lowercase should not match
        print_status("Test", "info")
        output = self.held_output.getvalue().strip()
        
        # Should use default bullet (not INFO symbol)
        self.assertEqual(output, "[14:30:45] • Test")
    
    @patch('src.analysis.io.datetime')
    def test_midnight_time(self, mock_datetime):
        """Test status at midnight."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 0, 0, 0)
        
        print_status("Midnight test", "INFO")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[00:00:00] ℹ Midnight test")
    
    @patch('src.analysis.io.datetime')
    def test_time_formatting(self, mock_datetime):
        """Test that time is properly zero-padded."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 9, 5, 3)
        
        print_status("Time format test", "INFO")
        output = self.held_output.getvalue().strip()
        
        self.assertEqual(output, "[09:05:03] ℹ Time format test")


class TestOutputIntegration(unittest.TestCase):
    """Integration tests for combined use of output functions."""
    
    def setUp(self):
        """Redirect stdout to capture print output."""
        self.held_output = StringIO()
        sys.stdout = self.held_output
    
    def tearDown(self):
        """Restore stdout."""
        sys.stdout = sys.__stdout__
    
    @patch('src.analysis.io.datetime')
    def test_typical_usage_pattern(self, mock_datetime):
        """Test a typical usage pattern as seen in analysis scripts."""
        mock_datetime.now.return_value = datetime(2026, 2, 27, 14, 30, 45)
        
        print_section_header("TEST ANALYSIS", "=")
        print_status("Started analysis", "INFO")
        print_subsection("Configuration")
        print_status("Analysis complete", "SUCCESS")
        
        output = self.held_output.getvalue()
        
        # Check all components are present
        self.assertIn("TEST ANALYSIS", output)
        self.assertIn("[14:30:45] ℹ Started analysis", output)
        self.assertIn("Configuration", output)
        self.assertIn("[14:30:45] ✓ Analysis complete", output)
        self.assertIn("=" * 80, output)
        self.assertIn("─" * 80, output)


if __name__ == '__main__':
    unittest.main()
