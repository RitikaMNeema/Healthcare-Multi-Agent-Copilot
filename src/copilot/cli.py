"""Local development/ops CLI - NOT the system's authorization boundary.

This talks to the graph and its SQLite files directly, with no resolved
API-key identity - `--user`/`--role`/`--approver` are trusted, unchecked
strings, the same way a `psql` shell trusts whoever is typing into it.
Someone with shell access to run this already has direct access to the
underlying database files, so no amount of in-CLI role-checking would add a
real boundary here (they could just edit the files, or import the modules
directly in a Python shell) - the actual authorization boundary is
`api/server.py`, which resolves a caller to a fixed identity from a server-
held API key and applies reviewer-role and separation-of-duties checks that
mean something precisely because the caller can't just... not go through it.

Because of that gap, the approval-deciding commands below (`approve`,
`pending`) refuse to run unless `COPILOT_ALLOW_CLI_APPROVALS` is explicitly
set truthy - a container running only `uvicorn api.server:app` (see
Dockerfile) never sets it, so `docker exec`-ing in and running
`copilot approve --approver anyone --decision approve` does nothing by
default; a local developer sets the env var once to use these commands. This
doesn't add real authorization (nothing here can, per above) - it just means
the bypass has to be a deliberate, one-time opt-in rather than something that
works out of the box in whatever environment this module happens to be
importable in.
"""
import argparse
import json
import os

from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.graph import build_graph, resume_request, run_request

_ALLOW_CLI_APPROVALS_ENV = "COPILOT_ALLOW_CLI_APPROVALS"


def _require_cli_approvals_enabled() -> None:
    if os.environ.get(_ALLOW_CLI_APPROVALS_ENV, "").lower() not in ("1", "true", "yes"):
        raise SystemExit(
            f"Refusing to run: this CLI is a trusted local-dev interface with no reviewer-role or "
            f"separation-of-duties checks (see api/server.py for those). Set "
            f"{_ALLOW_CLI_APPROVALS_ENV}=1 to use approval commands locally; a production deployment "
            f"should use the API server for approvals instead.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="send a request to the copilot")
    p_chat.add_argument("--query", required=True)
    p_chat.add_argument("--role", default="operator", choices=["viewer", "operator", "admin", "compliance_officer"])
    p_chat.add_argument("--user", default="local-user")
    p_chat.add_argument("--request-id")

    p_approve = sub.add_parser(
        "approve", help=f"resolve a pending approval (requires {_ALLOW_CLI_APPROVALS_ENV}=1 - see module docstring)",
    )
    p_approve.add_argument("--request-id", required=True)
    p_approve.add_argument("--approver", required=True)
    p_approve.add_argument("--decision", required=True, choices=["approve", "reject"])

    p_audit = sub.add_parser("audit", help="show the audit trail for a request")
    p_audit.add_argument("--request-id", required=True)

    sub.add_parser(
        "pending", help=f"list requests awaiting human approval (requires {_ALLOW_CLI_APPROVALS_ENV}=1)",
    )

    args = parser.parse_args()
    app = build_graph()

    if args.command == "chat":
        request_id, result = run_request(app, query=args.query, user_id=args.user, role=args.role, request_id=args.request_id)
        if "__interrupt__" in result:
            print(f"Request {request_id} is PENDING HUMAN APPROVAL.")
            print(f"Draft answer: {result.get('draft_answer')}")
            print(f"Risk: {result.get('guardrail_risk')}  Issues: {result.get('guardrail_issues')}")
            print(f"Approve with: copilot approve --request-id {request_id} --approver <you> --decision approve")
        else:
            print(f"Request {request_id}")
            print(result.get("final_answer"))

    elif args.command == "approve":
        _require_cli_approvals_enabled()
        result = resume_request(
            app, request_id=args.request_id, approved=(args.decision == "approve"), approver=args.approver,
        )
        print(result.get("final_answer"))

    elif args.command == "audit":
        for event in AuditLog().trail_for(args.request_id):
            print(f"[{event['event_type']}] {json.dumps(event['payload'])}")

    elif args.command == "pending":
        _require_cli_approvals_enabled()
        for pending in ApprovalQueue().list_pending():
            print(pending)


if __name__ == "__main__":
    main()
