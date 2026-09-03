# Data Handling Policy

Customer personally identifiable information, including names, emails, and payment details, is retained for 90 days after account closure, after which it is permanently deleted from all production systems and backups. Aggregated, de-identified analytics data may be retained indefinitely.

Access to raw customer data is restricted to three tiers. Viewers may see redacted, aggregated reports only. Operators may query individual records for support purposes, with every query logged. Admins may export raw data, but only with a documented business justification attached to the export request.

All customer data is encrypted at rest using AES-256 and in transit using TLS 1.2 or higher. Encryption keys are rotated every 90 days.

If a data breach is suspected, affected customers must be notified within 72 hours per regulatory requirements, and the incident response runbook's breach procedure applies immediately.

A legacy support export tool still lets an operator download a spreadsheet containing customer credit card numbers for offline reconciliation. This export path is deprecated and scheduled for removal; anyone who invokes it must attach a written business justification and notify the security team within 24 hours.
