# Edge Container Composer
# Dynamically spawns page containers based on visitor context

SHELL := /bin/bash
.PHONY: compose warm-funnel health clean

# Container registry (override with your registry)
REGISTRY ?= pages.local
REACT_BASE_IMAGE ?= node:20-alpine

# Context variables (passed from composer service)
SITE_ID ?= default
DEVICE ?= desktop
GEO ?= us
VARIANT ?= a
FUNNEL_DEPTH ?= 4

# Container naming
CONTAINER_PREFIX := page_$(SITE_ID)_$(DEVICE)_$(GEO)

# =============================================================================
# MAIN TARGETS
# =============================================================================

# Compose initial landing page container based on context
compose:
	@echo '{"action": "compose", "site_id": "$(SITE_ID)", "device": "$(DEVICE)", "geo": "$(GEO)", "variant": "$(VARIANT)"}'
	@docker run -d \
		--name $(CONTAINER_PREFIX)_$(VARIANT)_step1 \
		--label site_id=$(SITE_ID) \
		--label device=$(DEVICE) \
		--label geo=$(GEO) \
		--label variant=$(VARIANT) \
		--label funnel_step=1 \
		--network adaptive-ads-router_default \
		-e SITE_ID=$(SITE_ID) \
		-e DEVICE=$(DEVICE) \
		-e GEO=$(GEO) \
		-e VARIANT=$(VARIANT) \
		-e FUNNEL_STEP=1 \
		$(REGISTRY)/$(SITE_ID):$(VARIANT)-$(DEVICE)-$(GEO) 2>/dev/null || \
		docker start $(CONTAINER_PREFIX)_$(VARIANT)_step1 2>/dev/null || \
		echo '{"status": "already_running", "container": "$(CONTAINER_PREFIX)_$(VARIANT)_step1"}'
	@echo '{"status": "composed", "container": "$(CONTAINER_PREFIX)_$(VARIANT)_step1"}'

# Pre-warm the entire funnel (assumes non-bounce)
warm-funnel:
	@echo '{"action": "warm_funnel", "site_id": "$(SITE_ID)", "depth": $(FUNNEL_DEPTH)}'
	@for step in $$(seq 1 $(FUNNEL_DEPTH)); do \
		docker run -d \
			--name $(CONTAINER_PREFIX)_$(VARIANT)_step$${step} \
			--label site_id=$(SITE_ID) \
			--label funnel_step=$${step} \
			--network adaptive-ads-router_default \
			-e FUNNEL_STEP=$${step} \
			$(REGISTRY)/$(SITE_ID):$(VARIANT)-step$${step} 2>/dev/null || true; \
	done
	@echo '{"status": "funnel_warmed", "steps": $(FUNNEL_DEPTH)}'

# Compose specific funnel step
step-%:
	@echo '{"action": "compose_step", "step": $*, "site_id": "$(SITE_ID)"}'
	@docker run -d \
		--name $(CONTAINER_PREFIX)_$(VARIANT)_step$* \
		--label site_id=$(SITE_ID) \
		--label funnel_step=$* \
		--network adaptive-ads-router_default \
		-e FUNNEL_STEP=$* \
		$(REGISTRY)/$(SITE_ID):$(VARIANT)-step$* 2>/dev/null || \
		docker start $(CONTAINER_PREFIX)_$(VARIANT)_step$* 2>/dev/null || true
	@echo '{"status": "step_composed", "step": $*}'

# =============================================================================
# VARIANT MANAGEMENT
# =============================================================================

# Compose both A/B variants for split test
split-test:
	@$(MAKE) compose VARIANT=a
	@$(MAKE) compose VARIANT=b
	@echo '{"status": "split_test_ready", "variants": ["a", "b"]}'

# Swap to winning variant (stop loser, keep winner)
promote-winner:
	@echo '{"action": "promote", "winner": "$(VARIANT)", "site_id": "$(SITE_ID)"}'
	@docker stop $(CONTAINER_PREFIX)_$$([ "$(VARIANT)" = "a" ] && echo "b" || echo "a")_step1 2>/dev/null || true
	@docker rm $(CONTAINER_PREFIX)_$$([ "$(VARIANT)" = "a" ] && echo "b" || echo "a")_step1 2>/dev/null || true
	@echo '{"status": "winner_promoted", "variant": "$(VARIANT)"}'

# =============================================================================
# DEVICE/GEO SPECIFIC BUILDS
# =============================================================================

# Mobile-optimized container
mobile:
	@$(MAKE) compose DEVICE=mobile

# Desktop-optimized container
desktop:
	@$(MAKE) compose DEVICE=desktop

# Tablet-optimized container
tablet:
	@$(MAKE) compose DEVICE=tablet

# Region-specific builds
geo-%:
	@$(MAKE) compose GEO=$*

# =============================================================================
# NEURAL TRAINING EXPORT
# =============================================================================

# Export funnel state as JSON for AI accelerator
export-state:
	@echo '{'
	@echo '  "site_id": "$(SITE_ID)",'
	@echo '  "context": {'
	@echo '    "device": "$(DEVICE)",'
	@echo '    "geo": "$(GEO)",'
	@echo '    "variant": "$(VARIANT)"'
	@echo '  },'
	@echo '  "containers": ['
	@docker ps --filter "label=site_id=$(SITE_ID)" --format '    {"id": "{{.ID}}", "name": "{{.Names}}", "status": "{{.Status}}"},' | sed '$$ s/,$$//'
	@echo '  ],'
	@echo '  "timestamp": "'$$(date -Iseconds)'"'
	@echo '}'

# =============================================================================
# CLEANUP
# =============================================================================

# Stop all containers for a site
stop:
	@docker stop $$(docker ps -q --filter "label=site_id=$(SITE_ID)") 2>/dev/null || true
	@echo '{"status": "stopped", "site_id": "$(SITE_ID)"}'

# Remove all containers for a site
clean:
	@docker rm -f $$(docker ps -aq --filter "label=site_id=$(SITE_ID)") 2>/dev/null || true
	@echo '{"status": "cleaned", "site_id": "$(SITE_ID)"}'

# Health check
health:
	@echo '{"status": "healthy", "service": "edge-composer"}'
