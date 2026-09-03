# Deployment Policy

All production deployments require two independent approvals before they can be merged and released: one from the code owner and one from an on-call engineer who was not the author of the change. Deployments without two approvals will be automatically blocked by the release pipeline.

Change freeze windows apply during the last three business days of each fiscal quarter and during any active Sev1 or Sev2 incident. Emergency fixes during a freeze window still require two approvals, but one of them may be granted verbally by an engineering manager and recorded after the fact.

Every production deployment must have a documented rollback procedure that can be executed in under ten minutes. If a deployment causes an error-rate increase of more than 2% for five consecutive minutes, the on-call engineer must roll it back immediately rather than attempting a forward fix.

Deployment approvals, timestamps, and rollback actions are all written to the immutable deployment audit log, which is retained for seven years to satisfy compliance requirements.
