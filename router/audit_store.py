"""Durable audit storage for report payloads and delivery timelines."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, Any

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    select,
    desc,
    and_,
    inspect,
)


class AuditStore:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()

        self.payload_snapshots = Table(
            "report_payload_snapshots",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("site_id", String(128), nullable=False, index=True),
            Column("week_id", String(32), nullable=False),
            Column("mode", String(32), nullable=False),
            Column("report_email", String(320), nullable=False),
            Column("subject", Text, nullable=False),
            Column("html_body", Text, nullable=False),
            Column("text_body", Text, nullable=False),
            Column("payload_hash", String(64), nullable=False, index=True),
            Column("generated_at", DateTime, nullable=False),
            Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
        )

        self.delivery_events = Table(
            "report_delivery_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("site_id", String(128), nullable=False, index=True),
            Column("status", String(64), nullable=False, index=True),
            Column("mode", String(64), nullable=True),
            Column("week_id", String(32), nullable=True),
            Column("report_email", String(320), nullable=True),
            Column("provider", String(64), nullable=True),
            Column("provider_message_id", String(256), nullable=True, index=True),
            Column("provider_status", String(64), nullable=True),
            Column("payload_hash", String(64), nullable=True, index=True),
            Column("actor", String(128), nullable=True),
            Column("detail", JSON, nullable=True),
            Column("created_at", DateTime, nullable=False, default=datetime.utcnow, index=True),
        )

        self.provider_messages = Table(
            "report_provider_messages",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("provider", String(64), nullable=False, index=True),
            Column("provider_message_id", String(256), nullable=False, index=True),
            Column("site_id", String(128), nullable=False, index=True),
            Column("week_id", String(32), nullable=True),
            Column("report_email", String(320), nullable=True),
            Column("mode", String(64), nullable=True),
            Column("payload_hash", String(64), nullable=True),
            Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
        )

    def init(self) -> None:
        self.metadata.create_all(self.engine)

    def verify_schema(self) -> None:
        inspector = inspect(self.engine)
        required = {
            "report_payload_snapshots",
            "report_delivery_events",
            "report_provider_messages",
        }
        existing = set(inspector.get_table_names())
        missing = required - existing
        if missing:
            raise RuntimeError(f"Audit schema missing required tables: {sorted(missing)}")

    def save_payload_snapshot(self, payload: dict[str, Any]) -> None:
        generated_at = datetime.fromisoformat(payload["generated_at"])
        row = {
            "site_id": payload["site_id"],
            "week_id": payload["week_id"],
            "mode": payload.get("mode", "scheduled"),
            "report_email": payload["report_email"],
            "subject": payload["subject"],
            "html_body": payload["html_body"],
            "text_body": payload["text_body"],
            "payload_hash": payload["payload_hash"],
            "generated_at": generated_at,
            "created_at": datetime.utcnow(),
        }
        with self.engine.begin() as conn:
            conn.execute(self.payload_snapshots.insert().values(**row))

    def get_last_payload_snapshot(self, site_id: str) -> Optional[dict[str, Any]]:
        stmt = (
            select(self.payload_snapshots)
            .where(self.payload_snapshots.c.site_id == site_id)
            .order_by(desc(self.payload_snapshots.c.id))
            .limit(1)
        )
        with self.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if not row:
            return None
        return {
            "site_id": row["site_id"],
            "week_id": row["week_id"],
            "mode": row["mode"],
            "report_email": row["report_email"],
            "subject": row["subject"],
            "html_body": row["html_body"],
            "text_body": row["text_body"],
            "payload_hash": row["payload_hash"],
            "generated_at": row["generated_at"].isoformat(),
        }

    def append_delivery_event(self, event: dict[str, Any]) -> None:
        created_at = datetime.fromisoformat(event["timestamp"])
        detail = dict(event)
        for key in (
            "timestamp",
            "site_id",
            "status",
            "mode",
            "week_id",
            "report_email",
            "provider",
            "provider_message_id",
            "provider_status",
            "payload_hash",
            "actor",
        ):
            detail.pop(key, None)
        row = {
            "site_id": event["site_id"],
            "status": event["status"],
            "mode": event.get("mode"),
            "week_id": event.get("week_id"),
            "report_email": event.get("report_email"),
            "provider": event.get("provider"),
            "provider_message_id": event.get("provider_message_id"),
            "provider_status": event.get("provider_status"),
            "payload_hash": event.get("payload_hash"),
            "actor": event.get("actor"),
            "detail": detail or None,
            "created_at": created_at,
        }
        with self.engine.begin() as conn:
            conn.execute(self.delivery_events.insert().values(**row))

    def list_delivery_events(self, site_id: str, limit: int = 20) -> list[dict[str, Any]]:
        stmt = (
            select(self.delivery_events)
            .where(self.delivery_events.c.site_id == site_id)
            .order_by(desc(self.delivery_events.c.id))
            .limit(limit)
        )
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        events = []
        for row in rows:
            detail = row["detail"] or {}
            event = {
                "timestamp": row["created_at"].isoformat(),
                "site_id": row["site_id"],
                "status": row["status"],
                "mode": row["mode"],
                "week_id": row["week_id"],
                "report_email": row["report_email"],
                "provider": row["provider"],
                "provider_message_id": row["provider_message_id"],
                "provider_status": row["provider_status"],
                "payload_hash": row["payload_hash"],
                "actor": row["actor"],
            }
            event.update(detail)
            events.append(event)
        return events

    def bind_provider_message(self, provider: str, message_id: str, payload: dict[str, Any]) -> None:
        row = {
            "provider": provider,
            "provider_message_id": message_id,
            "site_id": payload["site_id"],
            "week_id": payload.get("week_id"),
            "report_email": payload.get("report_email"),
            "mode": payload.get("mode"),
            "payload_hash": payload.get("payload_hash"),
            "created_at": datetime.utcnow(),
        }
        with self.engine.begin() as conn:
            conn.execute(self.provider_messages.insert().values(**row))

    def lookup_provider_message(self, provider: str, message_id: str) -> Optional[dict[str, Any]]:
        stmt = (
            select(self.provider_messages)
            .where(
                and_(
                    self.provider_messages.c.provider == provider,
                    self.provider_messages.c.provider_message_id == message_id,
                )
            )
            .order_by(desc(self.provider_messages.c.id))
            .limit(1)
        )
        with self.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if not row:
            return None
        return {
            "site_id": row["site_id"],
            "week_id": row["week_id"],
            "report_email": row["report_email"],
            "mode": row["mode"],
            "payload_hash": row["payload_hash"],
        }
