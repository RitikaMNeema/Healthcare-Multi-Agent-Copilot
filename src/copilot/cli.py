import argparse
import json

from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.graph import build_graph, resume_request, run_request


def main() -> None:
    parser = argparse.ArgumentParser(prog="copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="send a request to the copilot")
    p_chat.add_argument("--query", required=True)
    p_chat.add_argument("--role", default="operator", choices=["viewer", "operator", "admin"])
    p_chat.add_argument("--user", default="local-user")
    p_chat.add_argument("--request-id")

    p_approve = sub.add_parser("approve", help="resolve a pending human-in-the-loop approval")
    p_approve.add_argument("--request-id", required=True)
    p_approve.add_argument("--approver", required=True)
    p_approve.add_argument("--decision", required=True, choices=["approve", "reject"])

    p_audit = sub.add_parser("audit", help="show the audit trail for a request")
    p_audit.add_argument("--request-id", required=True)

    sub.add_parser("pending", help="list requests awaiting human approval")

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
        result = resume_request(
            app, request_id=args.request_id, approved=(args.decision == "approve"), approver=args.approver,
        )
        print(result.get("final_answer"))

    elif args.command == "audit":
        for event in AuditLog().trail_for(args.request_id):
            print(f"[{event['event_type']}] {json.dumps(event['payload'])}")

    elif args.command == "pending":
        for pending in ApprovalQueue().list_pending():
            print(pending)


if __name__ == "__main__":
    main()
