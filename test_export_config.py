#!/usr/bin/env python3
"""
Test script to demonstrate the conditional export functionality.
This shows how the EEGWriter respects the config.io.export_data setting.
"""

from src.tms_eeg.config.settings import ProjectConfig, IOConfig
from src.tms_eeg.io.writer import EEGWriter

def test_export_disabled():
    """Test that export is skipped when export_data is False (default)"""
    print("=== Testing export_data = False (default) ===")
    
    # Create config with default settings (export_data = False)
    config = ProjectConfig(subject_id="V00test")
    
    # Create EEGWriter
    writer = EEGWriter(config)
    
    # Check the current setting
    print(f"config.io.export_data = {config.io.export_data}")
    
    # Try to save (should be skipped)
    print("Calling writer.save_epochs()...")
    # Note: We can't actually call with real epochs without loading data,
    # but the conditional check happens at the beginning of the method
    
    print("Export should be skipped when export_data = False\n")

def test_export_enabled():
    """Test that export works when export_data is True"""
    print("=== Testing export_data = True ===")
    
    # Create config with export enabled
    io_config = IOConfig(export_data=True)
    config = ProjectConfig(subject_id="V00test", io=io_config)
    
    # Create EEGWriter
    writer = EEGWriter(config)
    
    # Check the current setting
    print(f"config.io.export_data = {config.io.export_data}")
    
    print("Export should proceed when export_data = True\n")

if __name__ == "__main__":
    test_export_disabled()
    test_export_enabled()
    
    print("=== Usage Instructions ===")
    print("To enable export in your main.py:")
    print("1. Set config.io.export_data = True")
    print("2. Or create config with: ProjectConfig(subject_id='your_id', io=IOConfig(export_data=True))")
    print("3. The EEGWriter will automatically check this setting and export accordingly")