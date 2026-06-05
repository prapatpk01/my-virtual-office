# Virtual Office OpenClaw Integration Architecture

## Archive Note

This is a design-history artifact. It captures the product architecture goals
behind making Virtual Office portable across OpenClaw installations. It is not a
current implementation checklist.

## Document Purpose

This spec described the move from a single customized office into a portable,
plug-and-play product that can discover agents, visualize presence, route chat,
and persist office state through configurable product surfaces.

## Core Product Goals

- Discover agents from the connected runtime instead of relying on a fixed roster.
- Store office state in configurable data directories.
- Keep setup portable through the setup wizard and environment/config values.
- Show live presence through gateway/session activity.
- Keep manual presence APIs available for adapters and brokers.
- Make optional features feature-gated instead of hardwired to one deployment.

## Target Architecture

### Configuration Layer

Virtual Office should load user-editable settings from `vo-config.json`, then
allow environment variables to override those settings for container deployments.

Important configuration areas:

- office name and display settings
- data/status directory
- OpenClaw gateway URLs
- optional browser integration
- optional SMS/phone integration
- optional speech-to-text integration
- optional provider adapters

### Discovery Layer

Agent discovery should build the roster from runtime/provider APIs and product
configuration. The UI should render normalized office-agent records instead of
assuming a fixed set of agent IDs.

### Presence Layer

Presence should be derived from gateway/session activity where possible.
Explicit presence endpoints remain useful for provider adapters, manual
overrides, and brokers that represent work not visible through gateway events.

Relevant product surfaces:

- `GET /api/presence`
- `GET /status`
- `POST /api/presence/<agentId>`

### Office State Layer

Office layout, furniture, branches, meetings, projects, skills, and workflow
state should live under configured product storage rather than user-specific
paths.

### Setup Layer

The setup wizard should guide users through:

- connecting to OpenClaw
- naming the office
- detecting/importing agents
- configuring optional integrations
- entering or skipping a license key

### Delivery Layer

The public product should ship through Docker Compose and the published
container image. Local development details should stay out of public release
notes and current user documentation.

## Refactoring Themes

- Replace hardcoded rosters with runtime discovery.
- Replace fixed local paths with config/env values.
- Gate optional integrations behind setup/config.
- Keep public docs product-generic.
- Move historical implementation notes to `docs/design-history/`.

## Success Criteria

A fresh user should be able to:

1. Start the product with Docker Compose.
2. Open the setup wizard.
3. Connect a local OpenClaw runtime.
4. See detected agents in the office.
5. Use chat, presence, meetings, projects, and office editing without editing
   source code.

## Current Status

Many of the original portability goals are now implemented in the product. Use
current docs and code as the source of truth; this file is retained only for
architecture history.
