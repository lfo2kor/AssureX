# v2.0 Restart Checklist

**Purpose:** Quick checklist when restarting v2.0 implementation work

**Last Updated:** 2025-10-30

---

## 📋 PRE-RESTART CHECKLIST

### 1. Review Documentation (30 minutes)

- [ ] Read **spec_v2.0_with_feedback.md** (skim all 48 pages, focus on Sections 1-5)
- [ ] Read **v2.0_implementation_reference.md** (quick reference)
- [ ] Read **CHANGELOG_v2.0.md** (understand what's changed)
- [ ] Review this checklist completely

### 2. Understand Current System (15 minutes)

- [ ] Confirm v1.0 system is working
  ```bash
  python run_test.py --help
  ```
- [ ] Review 3 current agents:
  - [ ] `agents/jira_parser_agent.py`
  - [ ] `agents/vision_executor_agent.py`
  - [ ] `agents/report_generator_agent.py`
- [ ] Check `workflows/test_workflow.py` structure
- [ ] Understand `utils/step_executor.py` (3-level strategy)

### 3. Environment Verification (10 minutes)

- [ ] Python 3.11+ installed
  ```bash
  python --version
  ```
- [ ] Virtual environment activated
  ```bash
  .venv\Scripts\activate
  ```
- [ ] All dependencies installed
  ```bash
  pip list
  ```
- [ ] Environment variables set
  - [ ] `OPENAI_API_KEY`
- [ ] Playwright browsers installed
  ```bash
  playwright install
  ```

### 4. Test Current System (10 minutes)

- [ ] Run a test ticket to confirm v1.0 works
  ```bash
  python run_test.py RBPLCD-8835
  ```
- [ ] Verify outputs generated:
  - [ ] HTML report in `Reports/`
  - [ ] Playwright script in `Generated_Scripts/`
  - [ ] Video in `Videos/`
  - [ ] Screenshots in `Screenshots/`
  - [ ] Log in `Logs/`

---

## 🎯 IMPLEMENTATION PHASE CHECKLISTS

### Phase 1: Foundation (Week 1) ⏰ 5 days

**Goal:** Set up v2.0 structure without breaking v1.0

#### Day 1: Project Structure
- [ ] Create git branch `feature/feedback-agent-v2.0`
- [ ] Create new file: `agents/feedback_agent.py` (skeleton)
- [ ] Create new file: `utils/element_picker.py` (skeleton)
- [ ] Create new file: `utils/failure_analyzer.py` (skeleton)
- [ ] Update `.gitignore` for feedback files

#### Day 2: Configuration Schema
- [ ] Update `plcdtest_config.yaml` template
  - [ ] Add `execution.feedback_enabled: false` (default)
  - [ ] Add complete `feedback:` section
- [ ] Create `feedback_history.json` schema
- [ ] Create `feedback_rules.json` schema
- [ ] Create validation script: `utils/validate_config.py`

#### Day 3: State Dictionary Extension
- [ ] Update state TypedDict definition
  - [ ] Add `browser_session: dict`
  - [ ] Add `feedback_collected: bool`
  - [ ] Add `feedback_session: dict`
  - [ ] Add `retry_results: list[dict]`
- [ ] Document state changes in code comments

#### Day 4: Vision Executor Update
- [ ] Modify `agents/vision_executor_agent.py`
  - [ ] Add logic to keep browser open on failure
  - [ ] Add `browser_session` to state
  - [ ] Add check for `feedback_enabled` flag
- [ ] Test: Ensure v1.0 behavior unchanged when `feedback_enabled: false`

#### Day 5: Workflow Routing
- [ ] Update `workflows/test_workflow.py`
  - [ ] Add conditional node for feedback agent
  - [ ] Add routing logic: `has_failures and feedback_enabled`
  - [ ] Add edge: `vision_executor → feedback_agent → report_generator`
- [ ] Test: Confirm workflow still works in v1.0 mode

**Phase 1 Exit Criteria:**
- [ ] All v1.0 tests still pass
- [ ] New config flags present
- [ ] State dictionary extended
- [ ] Workflow routing added
- [ ] No breaking changes

---

### Phase 2: Element Picker (Week 2) ⏰ 5 days

**Goal:** Build visual selector tool

#### Day 1: JavaScript Overlay
- [ ] Implement `utils/element_picker.py`
  - [ ] Create JavaScript injection code
  - [ ] Add CSS styling for overlay
  - [ ] Add hover highlight effect
- [ ] Test: Inject into sample page, verify overlay appears

#### Day 2: Element Capture
- [ ] Add click event listener
- [ ] Capture clicked element
- [ ] Display element info panel (tagName, id, class, text)
- [ ] Test: Click elements, verify info captured

#### Day 3: Selector Extraction
- [ ] Extract 8 selector types:
  - [ ] data-testid
  - [ ] id
  - [ ] name
  - [ ] CSS selector path
  - [ ] XPath
  - [ ] Text content
  - [ ] Placeholder
  - [ ] aria-label
- [ ] Rank selectors by priority
- [ ] Test: Verify all selectors extracted correctly

#### Day 4: Validation
- [ ] Check selector uniqueness
- [ ] Verify element is interactable
- [ ] Test selector immediately on page
- [ ] Provide validation feedback to user
- [ ] Test: Try valid and invalid selections

#### Day 5: Integration Testing
- [ ] Integrate Element Picker with Playwright page
- [ ] Test with real test cases
- [ ] Handle edge cases (iframes, shadow DOM)
- [ ] Add timeout handling
- [ ] Test: Complete workflow with real failures

**Phase 2 Exit Criteria:**
- [ ] Element Picker works on test pages
- [ ] All selector types extracted
- [ ] Validation works correctly
- [ ] Handles edge cases gracefully
- [ ] Integrated with Playwright

---

### Phase 3: Failure Analysis (Week 3) ⏰ 5 days

**Goal:** Intelligent failure categorization

#### Day 1: Failure Categories
- [ ] Implement `utils/failure_analyzer.py`
- [ ] Define 6 category enums
- [ ] Create categorization rules
- [ ] Test: Categorize sample failures

#### Day 2: Diagnostics Collection
- [ ] Collect failure details:
  - [ ] Error message
  - [ ] Element state (visible, enabled, etc.)
  - [ ] Page state (URL, loaded, etc.)
  - [ ] Timing info
- [ ] Add to `step_executor.py`
- [ ] Test: Verify diagnostics captured

#### Day 3: Blocker Detection
- [ ] Implement dependency analysis
- [ ] Detect if step N failure causes step N+1 failure
- [ ] Categorize as blocker vs independent
- [ ] Test: Multi-step failure scenarios

#### Day 4: Failure Grouping
- [ ] Group related failures
- [ ] Prioritize blockers
- [ ] Create suggested fix order
- [ ] Test: Complex failure scenarios

#### Day 5: Integration
- [ ] Integrate with vision_executor_agent
- [ ] Add failure categorization to execution_results
- [ ] Update state with categorized failures
- [ ] Test: End-to-end failure analysis

**Phase 3 Exit Criteria:**
- [ ] All 6 categories implemented
- [ ] Blocker detection working
- [ ] Failure grouping accurate
- [ ] Integrated with executor
- [ ] Test coverage > 90%

---

### Phase 4: Feedback Collection (Week 4) ⏰ 5 days

**Goal:** Interactive correction workflow

#### Day 1: Feedback Agent Skeleton
- [ ] Implement main loop in `feedback_agent.py`
- [ ] Add state validation
- [ ] Add browser session check
- [ ] Test: Agent invoked correctly

#### Day 2: User Prompts
- [ ] Generate failure description for tester
- [ ] Display Element Picker instructions
- [ ] Add manual selector fallback prompt
- [ ] Test: Prompts are clear and helpful

#### Day 3: Element Picker Integration
- [ ] Activate Element Picker on failure
- [ ] Wait for user selection
- [ ] Capture selected element data
- [ ] Test: Full picker workflow

#### Day 4: Correction Validation
- [ ] Validate selector on live page
- [ ] Retry failed step immediately
- [ ] Show success/failure to tester
- [ ] Allow re-selection if validation fails
- [ ] Test: Valid and invalid selections

#### Day 5: Manual Fallback
- [ ] Implement text-based selector entry
- [ ] Validate manually entered selectors
- [ ] Test with DevTools-copied selectors
- [ ] Test: Complete workflow without picker

**Phase 4 Exit Criteria:**
- [ ] Feedback agent collects corrections
- [ ] Element Picker fully integrated
- [ ] Manual fallback works
- [ ] Validation prevents bad selectors
- [ ] User experience is smooth

---

### Phase 5: Configuration Updates (Week 5) ⏰ 5 days

**Goal:** Automatic learning system

#### Day 1: Selector Updates
- [ ] Implement `selectors.json` update logic
- [ ] Add new selectors from feedback
- [ ] Update existing selector metadata
- [ ] Backup before updates
- [ ] Test: File updated correctly

#### Day 2: Feedback History
- [ ] Log to `feedback_history.json`
- [ ] Record complete session data:
  - [ ] Timestamp, tester, ticket
  - [ ] Failed step details
  - [ ] Correction details
  - [ ] Retry results
- [ ] Test: Audit trail is complete

#### Day 3: Pattern Learning
- [ ] Implement `utils/pattern_learner.py`
- [ ] Detect row scoping patterns
- [ ] Detect timing adjustments
- [ ] Detect selector aliases
- [ ] Test: Patterns detected correctly

#### Day 4: Feedback Rules
- [ ] Write learned patterns to `feedback_rules.json`
- [ ] Add confidence scoring
- [ ] Track pattern application success
- [ ] Test: Rules applied on subsequent runs

#### Day 5: Integration
- [ ] Connect all update components
- [ ] Ensure atomic updates (all or nothing)
- [ ] Add rollback on failure
- [ ] Test: Complete update workflow

**Phase 5 Exit Criteria:**
- [ ] Selectors updated automatically
- [ ] Feedback history complete
- [ ] Patterns learned correctly
- [ ] Rules applied successfully
- [ ] Updates are atomic and safe

---

### Phase 6: Retry & Integration (Week 6) ⏰ 5 days

**Goal:** Close the feedback loop

#### Day 1: Failed Step Retry
- [ ] Implement retry logic with new selector
- [ ] Update execution_results with retry data
- [ ] Mark step as PASSED or FAILED (retry)
- [ ] Test: Single step retry

#### Day 2: Cascading Retry
- [ ] Retry dependent steps after blocker fix
- [ ] Track which steps were cascade retried
- [ ] Update execution_results
- [ ] Test: Multi-step cascade retry

#### Day 3: State Updates
- [ ] Update overall_status
- [ ] Update execution_results
- [ ] Add retry_results to state
- [ ] Test: State reflects retry accurately

#### Day 4: Workflow Integration
- [ ] Update `run_test.py` to call feedback_agent
- [ ] Ensure proper agent ordering
- [ ] Handle browser cleanup
- [ ] Test: Complete workflow end-to-end

#### Day 5: Report Integration
- [ ] Update `report_generator_agent.py`
- [ ] Show feedback session data in report
- [ ] Display retry results
- [ ] Highlight corrected steps
- [ ] Test: Report shows complete story

**Phase 6 Exit Criteria:**
- [ ] Retry logic works correctly
- [ ] Cascade retry successful
- [ ] State updated properly
- [ ] Workflow fully integrated
- [ ] Report shows feedback data

---

### Phase 7: Testing & Documentation (Week 7) ⏰ 5 days

**Goal:** Production readiness

#### Day 1: Unit Testing
- [ ] Test feedback_agent.py (all functions)
- [ ] Test element_picker.py
- [ ] Test failure_analyzer.py
- [ ] Test pattern_learner.py
- [ ] Target: 90%+ code coverage

#### Day 2: Integration Testing
- [ ] Test complete workflow with 10 test tickets
- [ ] Test with various failure types
- [ ] Test blocker detection
- [ ] Test pattern learning
- [ ] Test backward compatibility (v1.0 mode)

#### Day 3: User Acceptance Testing
- [ ] Invite QA team to test
- [ ] Collect feedback on UX
- [ ] Measure failure resolution time
- [ ] Verify < 5 minutes target met
- [ ] Fix critical issues

#### Day 4: Performance Testing
- [ ] Benchmark test execution time
- [ ] Verify no degradation vs v1.0
- [ ] Measure memory usage
- [ ] Measure disk usage
- [ ] Optimize if needed

#### Day 5: Documentation & Launch
- [ ] Update README.md
- [ ] Create user guide for testers
- [ ] Create admin guide for engineers
- [ ] Prepare launch announcement
- [ ] Deploy to production

**Phase 7 Exit Criteria:**
- [ ] All tests pass
- [ ] UAT successful
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Ready for production

---

## ✅ COMPLETION CHECKLIST

### Before Marking v2.0 Complete

- [ ] All 7 phases completed
- [ ] All exit criteria met
- [ ] Test execution time: 30-45s (maintained)
- [ ] Failure resolution time: < 5 minutes (achieved)
- [ ] Element Picker success rate: > 95%
- [ ] Pattern learning accuracy: > 80%
- [ ] Backward compatibility: 100% (verified)
- [ ] Code coverage: > 90%
- [ ] Documentation complete
- [ ] User training completed
- [ ] Production deployment successful

---

## 🔧 QUICK COMMANDS REFERENCE

### Development
```bash
# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run linting
flake8 agents/ utils/

# Run type checking
mypy agents/ utils/
```

### Testing
```bash
# Test v1.0 mode (no feedback)
python run_test.py RBPLCD-8835 --no-feedback

# Test v2.0 mode (with feedback)
python run_test.py RBPLCD-8835

# Test specific phase
python -m pytest tests/test_feedback_agent.py -v

# Test Element Picker standalone
python utils/test_element_picker.py
```

### Debugging
```bash
# Verbose logging
python run_test.py RBPLCD-8835 --log-level DEBUG

# Keep browser open for inspection
python run_test.py RBPLCD-8835 --no-cleanup --keep-browser

# Validate configuration
python utils/validate_config.py Configurations/MyApp/
```

---

## 📊 PROGRESS TRACKING

### Overall Progress

```
Phase 1: Foundation              [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 2: Element Picker          [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 3: Failure Analysis        [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 4: Feedback Collection     [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 5: Configuration Updates   [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 6: Retry & Integration     [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 7: Testing & Documentation [ ] Not Started  [ ] In Progress  [ ] Complete
```

### Key Milestones

- [ ] v2.0 specification complete (✅ DONE)
- [ ] Development environment set up
- [ ] Phase 1 complete (Foundation)
- [ ] Phase 2 complete (Element Picker)
- [ ] Phase 3 complete (Failure Analysis)
- [ ] Phase 4 complete (Feedback Collection)
- [ ] Phase 5 complete (Configuration Updates)
- [ ] Phase 6 complete (Retry & Integration)
- [ ] Phase 7 complete (Testing & Documentation)
- [ ] Production deployment
- [ ] v2.0 launch

---

## 🚨 CRITICAL REMINDERS

1. **Always test v1.0 compatibility** after each change
2. **Keep browser open** only when feedback_enabled and has_failures
3. **Validate selectors** before updating configuration files
4. **Backup configs** before making changes
5. **Complete audit trail** - log everything to feedback_history.json
6. **Pattern confidence** - don't apply low-confidence patterns
7. **User experience** - Element Picker must be intuitive
8. **Performance** - no degradation in execution time

---

## 📞 HELP RESOURCES

When stuck, check:
1. **spec_v2.0_with_feedback.md** - Complete reference (48 pages)
2. **v2.0_implementation_reference.md** - Quick reference (20 pages)
3. **CHANGELOG_v2.0.md** - What changed
4. **This checklist** - Step-by-step tasks

---

**WHEN RESTARTING:**
1. ✅ Read this checklist completely
2. ✅ Complete "Pre-Restart Checklist"
3. ✅ Choose phase to start/resume
4. ✅ Follow phase checklist
5. ✅ Check off items as you complete them

**Good luck with v2.0 implementation!**

---

**Last Updated:** 2025-10-30
**Status:** Ready to Start
**Estimated Timeline:** 7 weeks (35 business days)
