"""
PLCD Testing Assistant - Feedback Tool
Allows testers to correct wrong selectors and add contextual learning
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup

from config_loader import load_config, get_azure_client, get_embedding_model, get_chroma_client


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/feedback_tool.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeedbackTool:
    """
    Tool for collecting tester feedback and correcting learned selectors
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Feedback Tool

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.azure_client = get_azure_client(config)
        self.embedding_model = get_embedding_model(config)
        self.chroma_client = get_chroma_client(config)

        # Get runtime collection
        self.runtime_collection_name = config['vector_database']['collections']['runtime_learned']
        self.runtime_collection = self.chroma_client.get_or_create_collection(
            name=self.runtime_collection_name
        )

        logger.info("Feedback Tool initialized")


    def parse_html_report(self, report_path: str) -> Dict[str, Any]:
        """
        Parse HTML report to extract step information

        Args:
            report_path: Path to HTML report

        Returns:
            Dictionary containing ticket info and steps
        """
        logger.info(f"Parsing HTML report: {report_path}")

        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract ticket information
        container = soup.find('div', class_='container')
        if not container:
            raise Exception("Could not parse HTML report - invalid format")

        # Extract ticket ID and module from paragraph tags
        paragraphs = container.find_all('p')
        ticket_id = "Unknown"
        module = "Unknown"

        for p in paragraphs:
            text = p.get_text()
            if 'Ticket:' in text:
                # Extract ticket ID (format: "Ticket: RBPLCD-8835 - Title")
                ticket_id = text.split('Ticket:')[1].split('-')[0].strip() + '-' + text.split('-')[1].split()[0].strip()
            elif 'Module:' in text:
                module = text.split('Module:')[1].strip()

        # Extract step details from table
        step_table = soup.find('table', class_='step-table')
        steps = []

        if step_table:
            rows = step_table.find('tbody').find_all('tr')

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    step_num = cols[0].get_text().strip()
                    step_text = cols[1].get_text().strip()
                    selector = cols[2].find('span', class_='selector').get_text().strip()
                    agent = cols[3].find('span', class_='badge').get_text().strip()
                    confidence = float(cols[4].get_text().strip())
                    status = cols[5].find('span', class_='badge').get_text().strip()

                    steps.append({
                        'step_number': int(step_num),
                        'step_text': step_text,
                        'selector': selector,
                        'agent_used': agent,
                        'confidence': confidence,
                        'status': status
                    })

        logger.info(f"Parsed {len(steps)} steps from report")

        return {
            'ticket_id': ticket_id,
            'module': module,
            'steps': steps,
            'report_path': report_path
        }


    def normalize_selector(self, selector: str) -> str:
        """
        Normalize selector to valid CSS format
        Auto-adds brackets for attribute selectors if missing

        Args:
            selector: Raw selector input

        Returns:
            Normalized selector
        """
        selector = selector.strip()

        # If selector already has brackets or is a valid CSS selector, return as-is
        if selector.startswith('[') and selector.endswith(']'):
            return selector

        # Check for common CSS selector patterns that don't need brackets
        # (tag names, classes, IDs, pseudo-selectors, etc.)
        if (selector.startswith('.') or      # Class selector
            selector.startswith('#') or      # ID selector
            selector.startswith(':') or      # Pseudo-selector
            ' ' in selector or               # Combinator (space, >, +, ~)
            '>' in selector or
            '+' in selector or
            '~' in selector or
            not '=' in selector):            # No attribute assignment
            return selector

        # If it contains '=' but no brackets, it's likely an attribute selector
        # Auto-add brackets
        if '=' in selector and not selector.startswith('['):
            logger.info(f"Auto-fixing selector format: {selector} → [{selector}]")
            return f"[{selector}]"

        return selector


    def collect_feedback_interactive(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Interactively collect feedback from tester

        Args:
            report_data: Parsed report data

        Returns:
            List of corrections
        """
        print("\n" + "=" * 80)
        print("PLCD Testing Assistant - Feedback Tool")
        print("=" * 80)
        print(f"\nTicket: {report_data['ticket_id']}")
        print(f"Module: {report_data['module']}")
        print(f"Total Steps: {len(report_data['steps'])}")
        print("\n" + "-" * 80)
        print("Step Summary:")
        print("-" * 80)

        # Display all steps
        for step in report_data['steps']:
            status_icon = "✓" if step['status'] == 'PASSED' else "✗"
            print(f"{status_icon} Step {step['step_number']}: {step['step_text']}")
            print(f"   Selector: {step['selector']}")
            print(f"   Agent: {step['agent_used']} | Confidence: {step['confidence']:.2f} | Status: {step['status']}")
            print()

        # Ask which steps to correct
        print("-" * 80)
        step_input = input("Enter step numbers to correct (comma-separated, e.g., 4,5,6) or 'q' to quit: ").strip()

        if step_input.lower() == 'q':
            print("Exiting without changes.")
            return []

        # Parse step numbers
        try:
            step_numbers = [int(s.strip()) for s in step_input.split(',')]
        except ValueError:
            print("[ERROR] Invalid input. Please enter comma-separated numbers.")
            return []

        # Collect corrections
        corrections = []

        for step_num in step_numbers:
            # Find step
            step = next((s for s in report_data['steps'] if s['step_number'] == step_num), None)

            if not step:
                print(f"\n[WARNING] Step {step_num} not found in report. Skipping.")
                continue

            print("\n" + "=" * 80)
            print(f"Correcting Step {step_num}/{len(report_data['steps'])}")
            print("=" * 80)
            print(f"Step Text: {step['step_text']}")
            print(f"Current Selector: {step['selector']}")
            print(f"Current Agent: {step['agent_used']} | Confidence: {step['confidence']:.2f}")
            print(f"Status: {step['status']}")
            print("-" * 80)

            # Get correct selector
            correct_selector_raw = input("Enter correct selector: ").strip()

            if not correct_selector_raw:
                print("[WARNING] Empty selector. Skipping this step.")
                continue

            # Normalize selector format
            correct_selector = self.normalize_selector(correct_selector_raw)

            # Show normalized version if it was auto-fixed
            if correct_selector != correct_selector_raw:
                print(f"[AUTO-FIXED] Normalized to: {correct_selector}")

            # Get optional reason/context
            print("\nEnter reason/context (optional - helps AI learn better):")
            print("Examples:")
            print("  - 'Specific Parts accordion, not generic expansion panel'")
            print("  - 'Edit button icon, not the detail view container'")
            print("  - 'Type dropdown trigger, not the label'")
            reason = input("Reason: ").strip()

            # Confirm correction
            print("\n" + "-" * 80)
            print("Correction Summary:")
            print(f"  Step: {step['step_text']}")
            print(f"  Old Selector: {step['selector']}")
            print(f"  New Selector: {correct_selector}")
            print(f"  Reason: {reason if reason else '(none)'}")
            confirm = input("\nSave this correction? (y/n): ").strip().lower()

            if confirm == 'y':
                corrections.append({
                    'step_number': step_num,
                    'step_text': step['step_text'],
                    'old_selector': step['selector'],
                    'correct_selector': correct_selector,
                    'reason': reason,
                    'module': report_data['module'],
                    'ticket_id': report_data['ticket_id'],
                    'original_agent': step['agent_used'],
                    'original_confidence': step['confidence']
                })
                print("[OK] Correction saved.")
            else:
                print("[SKIPPED] Correction discarded.")

        return corrections


    def save_corrections_to_runtime(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save corrections to runtime collection

        Args:
            corrections: List of correction dictionaries

        Returns:
            Statistics about saved corrections
        """
        if not corrections:
            logger.info("No corrections to save")
            return {"corrections_saved": 0}

        logger.info("=" * 80)
        logger.info("Saving corrections to runtime collection")
        logger.info(f"Total corrections: {len(corrections)}")

        # Delete old wrong entries from runtime collection
        for correction in corrections:
            ticket_id = correction['ticket_id']
            step_num = correction['step_number']

            # Query to find existing entries for this step
            try:
                # Get all items from collection
                all_items = self.runtime_collection.get()

                # Find matching IDs (format: {ticket_id}_step_{step_num}_*)
                ids_to_delete = []
                if all_items['ids']:
                    for idx, item_id in enumerate(all_items['ids']):
                        metadata = all_items['metadatas'][idx]
                        if (metadata.get('ticket_id') == ticket_id and
                            metadata.get('step_number') == step_num):
                            ids_to_delete.append(item_id)

                # Delete old entries
                if ids_to_delete:
                    self.runtime_collection.delete(ids=ids_to_delete)
                    logger.info(f"Deleted {len(ids_to_delete)} old entries for Step {step_num}")

            except Exception as e:
                logger.warning(f"Could not delete old entries for Step {step_num}: {e}")

        # Prepare new entries with corrections
        entries_to_add = []
        embeddings_to_generate = []

        for correction in corrections:
            step_text = correction['step_text']
            correct_selector = correction['correct_selector']
            reason = correction['reason']

            # Create composite text for embedding
            # If reason provided, include it to enrich semantic context
            if reason:
                composite_text = f"{step_text} {reason}"
            else:
                composite_text = f"{step_text} {correct_selector}"

            # Create metadata (ensure all types are ChromaDB-compatible)
            metadata = {
                'module': str(correction['module']),
                'selector': str(correct_selector),
                'step_text': str(step_text),
                'ticket_id': str(correction['ticket_id']),
                'confidence': float(0.95),  # High confidence for user corrections
                'agent_used': 'UserCorrected',
                'learned_at': datetime.now().isoformat(),
                'step_number': int(correction['step_number']),
                'user_corrected': 'true',  # Store as string for ChromaDB compatibility
                'correction_reason': str(reason if reason else ''),
                'original_selector': str(correction['old_selector']),
                'original_agent': str(correction['original_agent']),
                'original_confidence': float(correction['original_confidence'])
            }

            # Create unique ID
            entry_id = f"{correction['ticket_id']}_step_{correction['step_number']}_corrected_{datetime.now().timestamp()}"

            entries_to_add.append({
                'id': entry_id,
                'metadata': metadata,
                'text': composite_text
            })

            embeddings_to_generate.append(composite_text)

        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(embeddings_to_generate)} corrections...")
        response = self.azure_client.embeddings.create(
            input=embeddings_to_generate,
            model=self.embedding_model
        )
        embeddings = [data.embedding for data in response.data]

        # Add to runtime collection
        ids = [entry['id'] for entry in entries_to_add]
        metadatas = [entry['metadata'] for entry in entries_to_add]
        documents = [entry['text'] for entry in entries_to_add]

        logger.info(f"Adding {len(ids)} corrected selectors to runtime collection...")
        self.runtime_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        logger.info(f"[OK] Saved {len(corrections)} corrections to runtime collection")
        logger.info("=" * 80)

        return {
            "corrections_saved": len(corrections),
            "collection": self.runtime_collection_name
        }


    def export_corrections_report(
        self,
        corrections: List[Dict[str, Any]],
        report_data: Dict[str, Any]
    ) -> str:
        """
        Export corrections to a JSON report for audit trail

        Args:
            corrections: List of corrections
            report_data: Original report data

        Returns:
            Path to exported report
        """
        # Create feedback folder
        feedback_folder = Path(self.config['folders'].get('feedback', 'Feedback'))
        feedback_folder.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_data['ticket_id']}_{timestamp}_feedback.json"
        filepath = feedback_folder / filename

        # Create report structure
        feedback_report = {
            "ticket_id": report_data['ticket_id'],
            "module": report_data['module'],
            "original_report": report_data['report_path'],
            "feedback_date": datetime.now().isoformat(),
            "corrections_count": len(corrections),
            "corrections": corrections
        }

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feedback_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported feedback report to: {filepath}")

        return str(filepath)


    def show_runtime_collection_stats(self):
        """Display statistics about runtime collection"""
        count = self.runtime_collection.count()

        print("\n" + "=" * 80)
        print("Runtime Collection Statistics")
        print("=" * 80)
        print(f"Total Selectors: {count}")

        if count > 0:
            all_items = self.runtime_collection.get()

            # Count by module
            modules = {}
            agents = {}
            user_corrected_count = 0

            for metadata in all_items['metadatas']:
                module = metadata.get('module', 'Unknown')
                agent = metadata.get('agent_used', 'Unknown')
                is_corrected = metadata.get('user_corrected', False)

                modules[module] = modules.get(module, 0) + 1
                agents[agent] = agents.get(agent, 0) + 1
                if is_corrected:
                    user_corrected_count += 1

            print(f"\nBy Module:")
            for module, count in modules.items():
                print(f"  {module}: {count}")

            print(f"\nBy Agent:")
            for agent, count in agents.items():
                print(f"  {agent}: {count}")

            print(f"\nUser Corrected: {user_corrected_count}")

        print("=" * 80)


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python feedback_tool.py <HTML_REPORT_PATH>")
        print("Example: python feedback_tool.py Reports\\RBPLCD-8835_20251118_103458_report.html")
        sys.exit(1)

    report_path = sys.argv[1]

    # Validate report path
    if not Path(report_path).exists():
        print(f"[ERROR] Report file not found: {report_path}")
        sys.exit(1)

    try:
        # Load configuration
        config = load_config()

        # Initialize feedback tool
        feedback_tool = FeedbackTool(config)

        # Show current stats
        feedback_tool.show_runtime_collection_stats()

        # Parse HTML report
        report_data = feedback_tool.parse_html_report(report_path)

        # Collect feedback interactively
        corrections = feedback_tool.collect_feedback_interactive(report_data)

        if not corrections:
            print("\n[INFO] No corrections provided. Exiting.")
            sys.exit(0)

        # Save corrections to runtime collection
        print("\n" + "=" * 80)
        print("Saving corrections to runtime collection...")
        print("=" * 80)

        stats = feedback_tool.save_corrections_to_runtime(corrections)

        print(f"\n[OK] Successfully saved {stats['corrections_saved']} corrections")

        # Export feedback report
        feedback_report_path = feedback_tool.export_corrections_report(corrections, report_data)
        print(f"[OK] Feedback report exported to: {feedback_report_path}")

        # Show updated stats
        feedback_tool.show_runtime_collection_stats()

        print("\n" + "=" * 80)
        print("Feedback collection complete!")
        print("=" * 80)
        print("\nNext time you run the same ticket, the corrected selectors will be used automatically.")
        print("Run the test again to verify the corrections:")
        print(f"  python plcd_ta.py {report_data['ticket_id']}\n")

    except Exception as e:
        print(f"\n[ERROR] Feedback tool failed: {e}")
        logger.error(f"Feedback tool failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
