#!/usr/bin/env python3
"""
MACHINE GAZE - Configuration Manager CLI

Tool for creating, validating, and managing configuration files.
Provides easy templates and configuration management.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_gaze.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_templates(manager: ConfigManager):
    """Print available templates."""
    templates = manager.list_templates()
    
    print("\n🎛️  CONFIGURATION TEMPLATES")
    print("=" * 50)
    for name, description in templates["config_templates"].items():
        print(f"  {name:20} - {description}")
    
    print("\n🔧 CLASSIFIER TEMPLATES")
    print("=" * 50)
    for name, description in templates["classifier_templates"].items():
        enabled = manager.classifier_templates[name].enabled
        status = "✅" if enabled else "❌"
        print(f"  {status} {name:20} - {description}")


def create_config_interactive(manager: ConfigManager):
    """Interactive configuration creation."""
    print("\n🎯 INTERACTIVE CONFIGURATION CREATOR")
    print("=" * 50)
    
    templates = manager.list_templates()
    
    # Select template
    print("\nAvailable configuration templates:")
    template_names = list(templates["config_templates"].keys())
    for i, (name, desc) in enumerate(templates["config_templates"].items(), 1):
        print(f"  {i}. {name} - {desc}")
    
    while True:
        try:
            choice = input(f"\nSelect template (1-{len(template_names)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(template_names):
                template_name = template_names[int(choice) - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return
    
    # Get output path
    default_name = f"{template_name}_config.yaml"
    output_path = input(f"\nOutput file path [{default_name}]: ").strip()
    if not output_path:
        output_path = default_name
    
    # Ask about modifications
    print(f"\nSelected template: {template_name}")
    modify = input("Would you like to modify any settings? (y/N): ").strip().lower()
    
    modifications = {}
    if modify in ['y', 'yes']:
        print("\n📝 Configuration Modifications")
        print("Enter modifications in key=value format (empty line to finish):")
        print("Examples:")
        print("  video_processor.overlay_only=true")
        print("  classifiers.yolo_world.confidence_threshold=0.7")
        print("  classifiers.face_emotion.enabled=true")
        
        while True:
            mod = input("Modification: ").strip()
            if not mod:
                break
            
            try:
                key, value = mod.split('=', 1)
                
                # Parse value
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                
                # Set nested keys
                keys = key.split('.')
                current = modifications
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                
                print(f"  ✅ Set {key} = {value}")
                
            except ValueError:
                print(f"  ❌ Invalid format: {mod}")
    
    # Create configuration
    success = manager.create_config_from_template(template_name, output_path, modifications)
    
    if success:
        print(f"\n✅ Configuration created: {output_path}")
        
        # Validate the created config
        validation = manager.validate_config(output_path)
        if validation["errors"]:
            print("⚠️  Configuration has errors:")
            for error in validation["errors"]:
                print(f"   - {error}")
        elif validation["warnings"]:
            print("⚠️  Configuration has warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        else:
            print("✅ Configuration is valid")
    else:
        print("❌ Failed to create configuration")


def validate_config_file(manager: ConfigManager, config_path: str):
    """Validate a configuration file and print results."""
    print(f"\n🔍 VALIDATING: {config_path}")
    print("=" * 50)
    
    if not Path(config_path).exists():
        print(f"❌ File not found: {config_path}")
        return
    
    validation = manager.validate_config(config_path)
    
    if validation["valid"]:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors")
    
    if validation["errors"]:
        print("\n🚨 ERRORS:")
        for error in validation["errors"]:
            print(f"   - {error}")
    
    if validation["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    
    if validation["suggestions"]:
        print("\n💡 SUGGESTIONS:")
        for suggestion in validation["suggestions"]:
            print(f"   - {suggestion}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="MACHINE GAZE - Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available templates
  python config_manager_cli.py list
  
  # Create config from template
  python config_manager_cli.py create emotion_analysis my_config.yaml
  
  # Interactive config creation
  python config_manager_cli.py interactive
  
  # Validate existing config
  python config_manager_cli.py validate config/my_config.yaml
  
  # Add classifier to existing config
  python config_manager_cli.py add-classifier config/my_config.yaml face_emotion
  
  # Export template documentation
  python config_manager_cli.py export-docs templates_guide.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List templates
    subparsers.add_parser('list', help='List available templates')
    
    # Create configuration
    create_parser = subparsers.add_parser('create', help='Create configuration from template')
    create_parser.add_argument('template', help='Template name')
    create_parser.add_argument('output', help='Output configuration file')
    create_parser.add_argument('--modify', '-m', action='append', 
                              help='Modifications (key=value format)')
    
    # Interactive creation
    subparsers.add_parser('interactive', help='Interactive configuration creation')
    
    # Validate configuration
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config', help='Configuration file to validate')
    
    # Add classifier
    add_parser = subparsers.add_parser('add-classifier', help='Add classifier to existing config')
    add_parser.add_argument('config', help='Configuration file to modify')
    add_parser.add_argument('classifier', help='Classifier name to add')
    add_parser.add_argument('--template', help='Classifier template to use')
    add_parser.add_argument('--param', action='append', 
                           help='Additional parameters (key=value format)')
    
    # Export documentation
    export_parser = subparsers.add_parser('export-docs', help='Export template documentation')
    export_parser.add_argument('output', help='Output documentation file')
    
    # Common arguments
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize manager
    manager = ConfigManager(args.config_dir)
    
    try:
        if args.command == 'list':
            print_templates(manager)
        
        elif args.command == 'create':
            modifications = {}
            if args.modify:
                for mod in args.modify:
                    try:
                        key, value = mod.split('=', 1)
                        # Parse value
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.replace('.', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                        
                        # Set nested keys
                        keys = key.split('.')
                        current = modifications
                        for k in keys[:-1]:
                            if k not in current:
                                current[k] = {}
                            current = current[k]
                        current[keys[-1]] = value
                    except ValueError:
                        print(f"Invalid modification format: {mod}")
                        sys.exit(1)
            
            success = manager.create_config_from_template(args.template, args.output, modifications)
            if success:
                print(f"✅ Created configuration: {args.output}")
            else:
                print("❌ Failed to create configuration")
                sys.exit(1)
        
        elif args.command == 'interactive':
            create_config_interactive(manager)
        
        elif args.command == 'validate':
            validate_config_file(manager, args.config)
        
        elif args.command == 'add-classifier':
            parameters = {}
            if args.param:
                for param in args.param:
                    try:
                        key, value = param.split('=', 1)
                        # Parse value
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.replace('.', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                        parameters[key] = value
                    except ValueError:
                        print(f"Invalid parameter format: {param}")
                        sys.exit(1)
            
            success = manager.add_classifier_to_config(
                args.config, args.classifier, args.template, parameters)
            if success:
                print(f"✅ Added classifier '{args.classifier}' to {args.config}")
            else:
                print("❌ Failed to add classifier")
                sys.exit(1)
        
        elif args.command == 'export-docs':
            success = manager.export_template_docs(args.output)
            if success:
                print(f"✅ Exported documentation to: {args.output}")
            else:
                print("❌ Failed to export documentation")
                sys.exit(1)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
