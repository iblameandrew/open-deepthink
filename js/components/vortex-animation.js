/**
 * VortexAnimation — ported from smenos/spirits/components/VortexAnimation.tsx
 *
 * Panel-scoped whirlpool (not fullscreen). Mounts into a container element.
 *
 * API:
 *   VortexAnimation.play({ container?, durationMs?, label? })
 *   VortexAnimation.stop()
 *   VortexAnimation.isPlaying()
 */
const VortexAnimation = (() => {
    'use strict';

    const DEFAULT_DURATION_MS = 12000;
    const PARTICLE_COUNT = 1600;
    const NOISE_DOTS = 2200;

    let rootEl = null;
    let canvasEl = null;
    let labelEl = null;
    let mountEl = null;
    let animationFrameId = null;
    let resizeHandler = null;
    let resizeObserver = null;
    let hideTimer = null;
    let playing = false;
    let startTime = 0;
    let totalDuration = DEFAULT_DURATION_MS;
    let particles = [];

    function resolveMount(options) {
        if (options.container instanceof HTMLElement) return options.container;
        if (typeof options.container === 'string') {
            const el = document.getElementById(options.container.replace(/^#/, ''));
            if (el) return el;
        }
        return (
            document.getElementById('qdad-vortex-panel')
            || document.getElementById('brainstorm-vortex-panel')
            || null
        );
    }

    function idleSelector() {
        return '.mode-vortex-idle, .qdad-vortex-idle';
    }

    function ensureDOM(mount) {
        if (!mount) return false;

        if (rootEl && mountEl === mount && mount.contains(rootEl)) {
            return true;
        }

        // Rebuild if mount target changed
        if (rootEl && rootEl.parentNode) {
            rootEl.parentNode.removeChild(rootEl);
        }

        mountEl = mount;
        rootEl = document.createElement('div');
        rootEl.className = 'vortex-panel';
        rootEl.setAttribute('aria-hidden', 'true');

        canvasEl = document.createElement('canvas');
        canvasEl.className = 'vortex-panel__canvas';

        const grain = document.createElement('div');
        grain.className = 'vortex-panel__grain';

        const scanlines = document.createElement('div');
        scanlines.className = 'vortex-panel__scanlines';

        labelEl = document.createElement('div');
        labelEl.className = 'vortex-panel__label';
        labelEl.innerHTML = '<span class="vortex-panel__label-text">Synthesizing</span>';

        rootEl.appendChild(canvasEl);
        rootEl.appendChild(grain);
        rootEl.appendChild(scanlines);
        rootEl.appendChild(labelEl);

        // Hide idle placeholder while vortex layer is present
        const idle = mount.querySelector(idleSelector());
        if (idle) idle.classList.add('is-hidden');

        mount.appendChild(rootEl);
        return true;
    }

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const isRed = Math.random() > 0.5;
            particles.push({
                radius: Math.random() * 280 + 16,
                angle: Math.random() * Math.PI * 2,
                baseSpeed: Math.random() * 0.005 + 0.001,
                size: Math.random() * 1.2 + 0.4,
                color: isRed ? '220, 38, 38' : '59, 130, 246',
            });
        }
    }

    function resizeCanvas(ctx) {
        if (!rootEl || !canvasEl) return;
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, rootEl.clientWidth || mountEl?.clientWidth || 320);
        const h = Math.max(1, rootEl.clientHeight || mountEl?.clientHeight || 320);
        canvasEl.width = Math.floor(w * dpr);
        canvasEl.height = Math.floor(h * dpr);
        canvasEl.style.width = w + 'px';
        canvasEl.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function renderFrame(ctx) {
        const now = performance.now();
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / totalDuration, 1);
        const speedMultiplier = 1 + progress * 4;

        const dpr = window.devicePixelRatio || 1;
        const width = canvasEl.width / dpr;
        const height = canvasEl.height / dpr;

        // 1. CLEAR & FLICKER
        const flicker = 1 - Math.random() * 0.15 * progress;
        ctx.fillStyle = `rgb(${Math.round(4 * flicker)}, ${Math.round(4 * flicker)}, ${Math.round(4 * flicker)})`;
        ctx.fillRect(0, 0, width, height);

        const centerX = width / 2;
        const centerY = height / 2;

        // 2. BACKGROUND FILM TEXTURE & HAIRS
        ctx.fillStyle = `rgba(255, 255, 255, ${0.1 + 0.2 * progress})`;
        for (let i = 0; i < NOISE_DOTS; i++) {
            const nx = Math.random() * width;
            const ny = Math.random() * height;
            const ns = Math.random() * 1.5;
            ctx.fillRect(nx, ny, ns, ns);
        }

        if (Math.random() < 0.4) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + 0.2 * progress})`;
            ctx.lineWidth = 0.5;
            const hairCount = Math.floor(Math.random() * 3) + 1;
            for (let h = 0; h < hairCount; h++) {
                const hx = Math.random() * width;
                const hy = Math.random() * height;
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                for (let seg = 0; seg < 5; seg++) {
                    ctx.lineTo(
                        hx + (Math.random() - 0.5) * 40,
                        hy + (Math.random() - 0.5) * 40
                    );
                }
                ctx.stroke();
            }
        }

        // 3. WHIRLPOOL PARTICLES
        const maxR = Math.min(width, height) * 0.55;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.angle += p.baseSpeed * speedMultiplier;
            const baseR = Math.min(p.radius, maxR);
            const currentRadius = baseR * (1 - progress * 0.5);
            const x = centerX + Math.cos(p.angle) * currentRadius;
            const y = centerY + Math.sin(p.angle) * currentRadius;

            const parts = p.color.split(',');
            const r = Math.round(parseInt(parts[0], 10) * (1 - progress) + 255 * progress);
            const g = Math.round(parseInt(parts[1], 10) * (1 - progress) + 255 * progress);
            const b = Math.round(parseInt(parts[2], 10) * (1 - progress) + 255 * progress);

            const alpha = (0.2 + Math.random() * 0.5) * (1 - progress * 0.3);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;

            ctx.beginPath();
            ctx.arc(x, y, p.size * (1 + progress * 2), 0, Math.PI * 2);
            ctx.fill();
        }

        // 4. CENTRAL BLOOM & OVERLAYS
        if (progress > 0.4) {
            const bloomProgress = (progress - 0.4) * 1.6;
            const bloomR = Math.min(width, height) * 0.35 * bloomProgress;
            const gradient = ctx.createRadialGradient(
                centerX, centerY, 0,
                centerX, centerY, bloomR
            );
            gradient.addColorStop(0, `rgba(255, 255, 255, ${0.4 * bloomProgress})`);
            gradient.addColorStop(0.5, `rgba(255, 255, 255, ${0.1 * bloomProgress})`);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);
        }

        // Vignette
        const vignette = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, Math.max(centerX, centerY) * 1.5
        );
        vignette.addColorStop(0, 'transparent');
        vignette.addColorStop(0.8, `rgba(0, 0, 0, ${0.3 * progress})`);
        vignette.addColorStop(1, `rgba(0, 0, 0, ${0.7 * progress})`);
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, width, height);

        if (progress < 1) {
            animationFrameId = requestAnimationFrame(() => renderFrame(ctx));
        } else {
            finish();
        }
    }

    function cleanupListeners() {
        if (animationFrameId != null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (resizeHandler) {
            window.removeEventListener('resize', resizeHandler);
            resizeHandler = null;
        }
        if (resizeObserver) {
            resizeObserver.disconnect();
            resizeObserver = null;
        }
        if (hideTimer) {
            clearTimeout(hideTimer);
            hideTimer = null;
        }
    }

    function showIdle() {
        if (!mountEl) return;
        const idle = mountEl.querySelector(idleSelector());
        if (idle) idle.classList.remove('is-hidden');
        if (rootEl) {
            rootEl.classList.remove('vortex-panel--active', 'vortex-panel--fading');
            rootEl.setAttribute('aria-hidden', 'true');
        }
    }

    function finish() {
        if (!playing) return;
        playing = false;
        cleanupListeners();

        if (rootEl) {
            rootEl.classList.add('vortex-panel--fading');
            hideTimer = setTimeout(() => {
                showIdle();
                hideTimer = null;
            }, 700);
        } else {
            showIdle();
        }
    }

    function stop() {
        playing = false;
        cleanupListeners();
        showIdle();
    }

    /**
     * Play the vortex inside a panel container.
     * @param {{ container?: string|HTMLElement, durationMs?: number, label?: string }} [opts]
     */
    function play(opts) {
        const options = opts || {};
        const mount = resolveMount(options);
        if (!mount) {
            console.warn('VortexAnimation: no mount container found (#qdad-vortex-panel)');
            return;
        }

        stop();
        if (!ensureDOM(mount)) return;

        totalDuration = options.durationMs > 0 ? options.durationMs : DEFAULT_DURATION_MS;
        if (labelEl) {
            const text = options.label || 'Synthesizing';
            labelEl.querySelector('.vortex-panel__label-text').textContent = text;
        }

        const idle = mount.querySelector(idleSelector());
        if (idle) idle.classList.add('is-hidden');

        rootEl.classList.remove('vortex-panel--fading');
        rootEl.classList.add('vortex-panel--active');
        rootEl.setAttribute('aria-hidden', 'false');

        const ctx = canvasEl.getContext('2d', { alpha: false });
        if (!ctx) return;

        resizeCanvas(ctx);
        resizeHandler = () => resizeCanvas(ctx);
        window.addEventListener('resize', resizeHandler);

        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(() => resizeCanvas(ctx));
            resizeObserver.observe(mount);
        }

        initParticles();
        startTime = performance.now();
        playing = true;
        animationFrameId = requestAnimationFrame(() => renderFrame(ctx));
    }

    function isPlaying() {
        return playing;
    }

    return { play, stop, isPlaying };
})();

if (typeof window !== 'undefined') {
    window.VortexAnimation = VortexAnimation;
}
