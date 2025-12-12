# Generated Playwright Test Scripts

This folder contains auto-generated Playwright test scripts from successful PLCD test executions.

## Prerequisites

```bash
pip install playwright pytest
playwright install
```

## Running Tests

### Run a specific test:
```bash
pytest RBPLCD-8835_20250117_120000_test.py -v -s
```

### Run all tests in folder:
```bash
pytest . -v -s
```

### Run with HTML report:
```bash
pytest . --html=report.html --self-contained-html
```

## Script Structure

Each generated script contains:
- **Auto-discovered selectors**: From Agent 1 (Semantic Search) or Agent 2 (DOM Discovery)
- **Confidence scores**: Indicating selector reliability
- **Comments**: Original test step descriptions
- **Actions**: Click, fill, select, verify operations

## Customization

Generated scripts are templates. You may need to:
1. Adjust input values (currently set to "test_value")
2. Add custom assertions for expected results
3. Modify timeouts for slower operations
4. Add additional verification steps

## Maintenance

- Scripts are timestamped to track generation time
- Each script is standalone and can be executed independently
- Update `conftest.py` to modify global browser settings
