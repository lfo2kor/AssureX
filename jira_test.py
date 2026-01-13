from jira import JIRA

jira = JIRA(
    server="https://rb-tracker.bosch.com/tracker03",
    token_auth="your_auth_token"
)

issue = jira.issue("RBPLCD-8554")

print("=" * 50)
print(f"Ticket ID: {issue.key}")
print(f"Title: {issue.fields.summary}")
print(f"Status: {issue.fields.status}")
print(f"Priority: {issue.fields.priority}")
print(f"Assignee: {issue.fields.assignee}")
print(f"Reporter: {issue.fields.reporter}")
print("=" * 50)
print("Description:")
print(issue.fields.description or "No description available")
print("=" * 50)