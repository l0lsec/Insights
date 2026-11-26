/*
 * window.Insights: the shared feedback pieces every page uses.
 *
 * Loaded once, from <head> in templates/base.html, before any page script, so a
 * page can call these while it is still parsing. Never load it again from a
 * page, and never copy one of these into a page: a second toast helper is how
 * compose.html and article.html ended up with two that disagree.
 *
 * The rules they carry out are docs/UX_PRINCIPLES.md:
 *   toast      say what happened at once; errors stay until dismissed; Undo/Retry (laws 6, 10, 15)
 *   confirm    a real dialog that names what will happen; Cancel first when it is risky (laws 14, 18)
 *   busy       a click shows a response at once and cannot fire twice (law 6)
 *   explain    plain words for a failure, never a raw error (law 15)
 *   fieldError the problem next to the field, with the input kept (laws 8, 15)
 *   deferred   Undo for a delete that stays inside Insights, without a soft delete (laws 15, 18)
 *
 * scripts/check_ux.py runs this file under node (mode "behavior") and fails if
 * any of the above stops being true.
 */
(function (global) {
    'use strict';
    if (global.Insights && global.Insights.__ui) return;   // loaded twice: keep the first

    var doc = global.document;
    var TYPES = { success: 1, info: 1, warning: 1, error: 1 };
    var LIFETIME = { success: 5000, info: 5000, warning: 8000, error: 0 };   // 0 = until dismissed
    var MAX_TOASTS = 4;

    function make(tag, className, text) {
        var node = doc.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = String(text);
        return node;
    }

    function button(label, className, onClick) {
        var b = make('button', className, label);
        b.setAttribute('type', 'button');
        b.addEventListener('click', onClick);
        return b;
    }

    /* ------------------------------------------------------------- toast */

    function stack() {
        var region = doc.getElementById('insightsToasts');
        if (!region) {
            region = make('div', 'insights-toasts');
            region.id = 'insightsToasts';
            region.setAttribute('aria-live', 'polite');
            doc.body.appendChild(region);
        }
        return region;
    }

    /**
     * Insights.toast(message, typeOrOptions)
     *   type      'success' | 'info' | 'warning' | 'error'   (default 'info')
     *   undo      function: adds an Undo button; clicking it runs the function
     *   action    {label, run}: one extra button, e.g. {label: 'Retry', run: save}
     *   duration  ms before it goes; 0 keeps it. Errors default to 0.
     * The message is text, never HTML. Returns {element, dismiss()}.
     */
    function toast(message, opts) {
        if (typeof opts === 'string') opts = { type: opts };
        opts = opts || {};
        var type = TYPES[opts.type] ? opts.type : 'info';
        var duration = opts.duration != null ? opts.duration
            : (opts.undo ? 6000 : (opts.sticky ? 0 : LIFETIME[type]));

        var region = stack();
        var item = make('div', 'insights-toast insights-toast-' + type);
        // Errors interrupt (role=alert); the rest are announced politely by the region.
        item.setAttribute('role', type === 'error' ? 'alert' : 'status');
        var body = make('div', 'insights-toast-body', message);
        item.appendChild(body);

        var timer = null, gone = false;
        function dismiss() {
            if (gone) return;
            gone = true;
            if (timer) global.clearTimeout(timer);
            if (item.parentNode) item.parentNode.removeChild(item);
        }
        function arm() {
            if (duration > 0 && !gone) timer = global.setTimeout(dismiss, duration);
        }
        function pause() { if (timer) { global.clearTimeout(timer); timer = null; } }

        var actions = make('div', 'insights-toast-actions');
        if (typeof opts.undo === 'function') {
            actions.appendChild(button('Undo', 'btn btn-sm btn-outline-primary insights-toast-undo', function () {
                dismiss();
                opts.undo();
            }));
        }
        if (opts.action && typeof opts.action.run === 'function') {
            actions.appendChild(button(opts.action.label || 'Retry', 'btn btn-sm btn-outline-primary insights-toast-action', function () {
                dismiss();
                opts.action.run();
            }));
        }
        var close = button('', 'btn-close insights-toast-close', dismiss);
        close.setAttribute('aria-label', 'Dismiss');
        actions.appendChild(close);
        item.appendChild(actions);

        // Someone reading or about to click should not have it vanish under them.
        item.addEventListener('mouseenter', pause);
        item.addEventListener('focusin', pause);
        item.addEventListener('mouseleave', arm);
        item.addEventListener('focusout', arm);

        region.appendChild(item);
        // Keep the stack short: drop the oldest one that is not an error first.
        var shown = region.querySelectorAll('.insights-toast');
        if (shown.length > MAX_TOASTS) {
            var victim = null;
            for (var i = 0; i < shown.length && !victim; i++) {
                if (shown[i] !== item && shown[i].getAttribute('role') !== 'alert') victim = shown[i];
            }
            victim = victim || shown[0];
            if (victim.parentNode) victim.parentNode.removeChild(victim);
        }
        arm();
        return { element: item, dismiss: dismiss };
    }

    /* ----------------------------------------------------------- confirm */

    var confirmQueue = Promise.resolve();
    var dialog = null;

    function buildDialog() {
        var modal = make('div', 'modal fade insights-confirm');
        modal.id = 'insightsConfirm';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-hidden', 'true');
        modal.setAttribute('aria-labelledby', 'insightsConfirmTitle');
        modal.setAttribute('aria-describedby', 'insightsConfirmBody');
        var box = make('div', 'modal-dialog modal-dialog-centered');
        var content = make('div', 'modal-content');
        var header = make('div', 'modal-header');
        var title = make('h2', 'modal-title fs-5');
        title.id = 'insightsConfirmTitle';
        header.appendChild(title);
        var body = make('div', 'modal-body insights-confirm-body');
        body.id = 'insightsConfirmBody';
        var footer = make('div', 'modal-footer');
        var cancel = make('button', 'btn btn-outline-secondary');
        cancel.setAttribute('type', 'button');
        cancel.setAttribute('data-insights-confirm', 'cancel');
        var ok = make('button', 'btn btn-primary');
        ok.setAttribute('type', 'button');
        ok.setAttribute('data-insights-confirm', 'ok');
        footer.appendChild(cancel);
        footer.appendChild(ok);
        content.appendChild(header);
        content.appendChild(body);
        content.appendChild(footer);
        box.appendChild(content);
        modal.appendChild(box);
        doc.body.appendChild(modal);
        return { modal: modal, title: title, body: body, cancel: cancel, ok: ok };
    }

    function askOnce(opts) {
        return new Promise(function (resolve) {
            var Modal = global.bootstrap && global.bootstrap.Modal;
            if (!Modal) {
                // No Bootstrap (a bare page): the browser's own dialog, same answer shape.
                resolve(global.confirm([opts.title, opts.body].filter(Boolean).join('\n\n')));
                return;
            }
            if (!dialog) dialog = buildDialog();
            var d = dialog;
            var returnFocus = doc.activeElement;
            d.title.textContent = opts.title || 'Are you sure?';
            d.body.textContent = opts.body || '';
            d.body.hidden = !opts.body;
            d.ok.textContent = opts.confirmLabel || 'Confirm';
            d.cancel.textContent = opts.cancelLabel || 'Cancel';
            d.ok.className = 'btn ' + (opts.danger ? 'btn-danger' : 'btn-primary');

            var answer = false;
            var instance = Modal.getOrCreateInstance(d.modal, { backdrop: true, keyboard: true, focus: true });
            function onOk() { answer = true; instance.hide(); }
            function onCancel() { answer = false; instance.hide(); }
            function onShown() {
                // Risky: the safe choice is the one Enter hits. Otherwise the main action.
                (opts.danger ? d.cancel : d.ok).focus();
            }
            function onHidden() {
                d.ok.removeEventListener('click', onOk);
                d.cancel.removeEventListener('click', onCancel);
                d.modal.removeEventListener('shown.bs.modal', onShown);
                d.modal.removeEventListener('hidden.bs.modal', onHidden);
                if (returnFocus && typeof returnFocus.focus === 'function') returnFocus.focus();
                resolve(answer);
            }
            d.ok.addEventListener('click', onOk);
            d.cancel.addEventListener('click', onCancel);
            d.modal.addEventListener('shown.bs.modal', onShown);
            d.modal.addEventListener('hidden.bs.modal', onHidden);
            instance.show();
        });
    }

    /**
     * Insights.confirm({title, body, confirmLabel, cancelLabel, danger}) -> Promise<boolean>
     * Say what will happen and to how many things, in the title or body
     * ("Delete 3 posts?", "They are removed from every platform's queue.").
     * Prefer Insights.deferred (Undo) for anything that stays inside Insights.
     * A second call while one is open waits its turn.
     */
    function confirm(opts) {
        if (typeof opts === 'string') opts = { title: opts };
        opts = opts || {};
        var turn = confirmQueue.then(function () { return askOnce(opts); });
        confirmQueue = turn.then(function () {}, function () {});
        return turn;
    }

    /* -------------------------------------------------------------- busy */

    var inFlight = typeof WeakMap === 'function' ? new WeakMap() : null;

    /**
     * Insights.busy(button, promiseOrFunction, {label}) -> the same promise
     * The button is disabled and says "Working..." (or label) at once, and comes
     * back as it was when the work settles, success or failure. A second click
     * while it is busy starts nothing and gets the promise already running.
     */
    function busy(btn, work, opts) {
        opts = opts || {};
        if (!btn) return Promise.resolve().then(function () { return typeof work === 'function' ? work() : work; });
        var running = inFlight && inFlight.get(btn);
        if (running) return running;

        var before = { html: btn.innerHTML, disabled: btn.disabled };
        btn.disabled = true;
        btn.setAttribute('aria-busy', 'true');
        btn.innerHTML = '';
        var spin = make('span', 'spinner-border spinner-border-sm me-2');
        spin.setAttribute('aria-hidden', 'true');
        btn.appendChild(spin);
        btn.appendChild(doc.createTextNode(opts.label || 'Working...'));

        var promise = Promise.resolve().then(function () {
            return typeof work === 'function' ? work() : work;
        });
        function restore() {
            if (inFlight) inFlight.delete(btn);
            btn.innerHTML = before.html;
            btn.disabled = before.disabled;
            btn.removeAttribute('aria-busy');
        }
        var settled = promise.then(function (value) { restore(); return value; },
                                   function (error) { restore(); throw error; });
        if (inFlight) inFlight.set(btn, settled);
        return settled;
    }

    /* ----------------------------------------------------------- explain */

    var TECHNICAL = /unexpected token|json|undefined|null\b|is not a function|is not defined|cannot read|typeerror|referenceerror|syntaxerror|networkerror|failed to fetch|load failed|traceback|exception|errno|\bat \S+:\d+/i;

    var BY_STATUS = {
        400: 'Insights could not accept that. Check what you entered and try again.',
        401: 'Your session ended. Sign in again, then retry.',
        403: 'You do not have permission to do that.',
        404: 'That item no longer exists. Reload the page to see the current list.',
        409: 'That changed somewhere else. Reload the page, then try again.',
        413: 'That file is too large to upload.',
        422: 'Insights could not accept that. Check what you entered and try again.',
        429: 'Too many requests at once. Wait a moment, then try again.'
    };
    var SERVER = 'Insights hit a problem on its side. Nothing you entered was lost; try again in a moment.';
    var OFFLINE = 'Could not reach Insights. Check your connection, then try again.';
    var UNKNOWN = 'Something went wrong. Nothing you entered was lost; try again.';

    /**
     * Insights.explain(errorOrResponse, context) -> a sentence or two for a person.
     * context says what failed ("Could not save the post."); the rest says why and
     * what to do. Accepts an Error, a fetch Response or {status}, a server's JSON
     * {error: "..."} or a plain string. A message that reads like code is replaced.
     * Set error.userMessage to pass your own wording through untouched.
     */
    function explain(err, context) {
        var why = reason(err);
        return context ? (String(context).replace(/\s*$/, '') + ' ' + why) : why;
    }

    function reason(err) {
        if (err == null) return UNKNOWN;
        if (typeof err === 'string') return TECHNICAL.test(err) ? UNKNOWN : err;
        if (err.userMessage) return String(err.userMessage);
        if (err.name === 'AbortError') return 'That was cancelled before it finished.';
        if (err.name === 'TypeError' && /fetch|network|load failed/i.test(err.message || '')) return OFFLINE;
        if (global.navigator && global.navigator.onLine === false) return OFFLINE;
        var status = typeof err.status === 'number' ? err.status : null;
        if (typeof err.error === 'string' && err.error && !TECHNICAL.test(err.error)) return err.error;
        if (status) return BY_STATUS[status] || (status >= 500 ? SERVER : UNKNOWN);
        if (err.message && !TECHNICAL.test(err.message) && err.message.length <= 300) return err.message;
        return UNKNOWN;
    }

    /* -------------------------------------------------------- fieldError */

    var fieldSeq = 0;

    /**
     * Insights.fieldError(input, message, {focus})
     * Shows the message under the field and marks it invalid; the value is left
     * alone. The next edit clears it. A blank message clears it now.
     */
    function fieldError(input, message, opts) {
        if (!input) return input;
        opts = opts || {};
        var holder = input.parentNode;
        var note = input.__insightsNote || null;
        if (!message) {
            input.classList.remove('is-invalid');
            input.removeAttribute('aria-invalid');
            if (note && note.parentNode) note.parentNode.removeChild(note);
            input.__insightsNote = null;
            return input;
        }
        if (!note) {
            note = make('div', 'invalid-feedback d-block');
            note.id = (input.id || 'insights-field-' + (++fieldSeq)) + '-error';
            if (holder) {
                if (input.nextSibling) holder.insertBefore(note, input.nextSibling);
                else holder.appendChild(note);
            }
            input.__insightsNote = note;
            var described = (input.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
            if (described.indexOf(note.id) === -1) described.push(note.id);
            input.setAttribute('aria-describedby', described.join(' '));
            input.addEventListener('input', function clear() {
                input.removeEventListener('input', clear);
                fieldError(input, '');
            });
        }
        note.textContent = message;
        input.classList.add('is-invalid');
        input.setAttribute('aria-invalid', 'true');
        if (opts.focus && typeof input.focus === 'function') input.focus();
        return input;
    }

    /* ---------------------------------------------------------- deferred */

    var pending = [];

    /**
     * Insights.deferred(run, {message, hide, restore, undoMs, failMessage})
     *   -> Promise<{undone} | {committed, value} | {failed, error}>
     *
     * Undo for an action that stays inside Insights (delete, clear, remove from
     * the queue). hide() runs at once, so the page answers the click; the toast
     * offers Undo for undoMs (6 s); only then does run({keepalive}) send the
     * request. Undo calls restore() and nothing is sent. If the page is left
     * first, the request goes out with keepalive, so pass it to fetch:
     *     run: function (o) { return fetch(url, {method: 'POST', keepalive: o.keepalive}) }
     * If run fails (rejects, or resolves a Response that is not ok), restore()
     * puts the item back and an error toast offers Retry.
     * Not for actions that leave Insights (publish, send, disconnect): confirm those.
     */
    function deferred(run, opts) {
        opts = opts || {};
        var undoMs = opts.undoMs != null ? opts.undoMs : 6000;
        var hide = opts.hide || function () {};
        var restore = opts.restore || function () {};

        return new Promise(function (resolve) {
            var state = 'waiting', timer = null, note = null;
            var entry = { flush: function () { commit(true); } };

            function settle(result) {
                var at = pending.indexOf(entry);
                if (at !== -1) pending.splice(at, 1);
                resolve(result);
            }
            function attempt(keepalive) {
                var result;
                try { result = Promise.resolve(run({ keepalive: !!keepalive })); }
                catch (error) { result = Promise.reject(error); }
                return result.then(function (value) {
                    if (value && typeof value.ok === 'boolean' && !value.ok) throw value;
                    return value;
                });
            }
            function fail(error) {
                restore();
                toast(explain(error, opts.failMessage || 'That did not go through, so it is back.'), {
                    type: 'error',
                    action: { label: 'Retry', run: function () {
                        hide();
                        attempt(false).then(function (value) {
                            settle({ committed: true, value: value });
                        }, function (again) { fail(again); });
                    } }
                });
                // A retry may still win; the first result is the one reported.
                settle({ failed: true, error: error });
            }
            function commit(keepalive) {
                if (state !== 'waiting') return;
                state = 'sent';
                if (timer) global.clearTimeout(timer);
                if (note) note.dismiss();
                attempt(keepalive).then(function (value) {
                    settle({ committed: true, value: value });
                }, fail);
            }
            function undo() {
                if (state !== 'waiting') return;
                state = 'undone';
                if (timer) global.clearTimeout(timer);
                restore();
                settle({ undone: true });
            }

            hide();
            pending.push(entry);
            note = toast(opts.message || 'Done.', { type: 'success', undo: undo, duration: 0 });
            timer = global.setTimeout(function () { commit(false); }, undoMs);
        });
    }

    function flushPending() {
        pending.slice().forEach(function (entry) { entry.flush(); });
    }
    if (global.addEventListener) global.addEventListener('pagehide', flushPending);

    /* ------------------------------------------------ declarative markup */

    /*
     * For server-rendered forms and links, so a template needs no script:
     *
     *   <form action="/feed/3/delete" method="post"
     *         data-undo="Deleted the feed Lex Fridman." data-undo-row="tr">
     *     Submits only after the Undo window; the closest <tr> (or the element
     *     matching data-undo-row, default [data-undo-row]) hides at once and comes
     *     back on Undo or failure. data-undo-then="reload" or a URL runs after it
     *     went through. Works on <a href data-undo> too (sent as GET).
     *
     *   <form ... data-confirm="Your token is deleted; queued posts stop."
     *         data-confirm-title="Disconnect LinkedIn?" data-confirm-label="Disconnect"
     *         data-confirm-danger>
     *     Asks with Insights.confirm first; also on <a> and <button>.
     */
    function attr(el, name) { return el.getAttribute(name); }

    function confirmOptsFrom(el) {
        return {
            title: attr(el, 'data-confirm-title') || attr(el, 'data-confirm'),
            body: attr(el, 'data-confirm-title') ? attr(el, 'data-confirm') : '',
            confirmLabel: attr(el, 'data-confirm-label') || 'Confirm',
            danger: el.hasAttribute('data-confirm-danger')
        };
    }

    function undoRow(el) {
        var sel = attr(el, 'data-undo-row');
        if (sel && el.closest) return el.closest(sel) || doc.querySelector(sel) || el;
        return (el.closest && el.closest('[data-undo-row]')) || el;
    }

    function undoFrom(el, send) {
        var row = undoRow(el);
        var then = attr(el, 'data-undo-then');
        return deferred(send, {
            message: attr(el, 'data-undo') || 'Deleted.',
            hide: function () { row.hidden = true; },
            restore: function () { row.hidden = false; }
        }).then(function (result) {
            if (result.committed && then) {
                if (then === 'reload') global.location.reload();
                else global.location.href = then;
            }
            return result;
        });
    }

    function sendForm(form, keepalive) {
        return global.fetch(form.action, {
            method: (form.getAttribute('method') || 'POST').toUpperCase(),
            body: new global.FormData(form),
            credentials: 'same-origin',
            keepalive: !!keepalive
        });
    }

    function onSubmit(event) {
        var form = event.target;
        if (!form || !form.hasAttribute) return;
        if (form.__insightsPass) { form.__insightsPass = false; return; }
        var wantsConfirm = form.hasAttribute('data-confirm');
        var wantsUndo = form.hasAttribute('data-undo');
        if (!wantsConfirm && !wantsUndo) return;
        event.preventDefault();
        var asked = wantsConfirm ? confirm(confirmOptsFrom(form)) : Promise.resolve(true);
        return asked.then(function (yes) {
            if (!yes) return false;
            if (wantsUndo) return undoFrom(form, function (o) { return sendForm(form, o.keepalive); });
            form.__insightsPass = true;
            if (typeof form.requestSubmit === 'function') form.requestSubmit();
            else form.submit();
            return true;
        });
    }

    function onClick(event) {
        var el = event.target && event.target.closest && event.target.closest('a[data-confirm], button[data-confirm], a[data-undo]');
        if (!el) return;
        if (el.__insightsPass) { el.__insightsPass = false; return; }
        event.preventDefault();
        if (event.stopImmediatePropagation) event.stopImmediatePropagation();
        var asked = el.hasAttribute('data-confirm') ? confirm(confirmOptsFrom(el)) : Promise.resolve(true);
        return asked.then(function (yes) {
            if (!yes) return false;
            if (el.hasAttribute('data-undo')) {
                return undoFrom(el, function (o) {
                    return global.fetch(el.href, { credentials: 'same-origin', keepalive: !!o.keepalive });
                });
            }
            el.__insightsPass = true;
            el.click();   // runs the element's own handler, or follows the link / submits its form
            return true;
        });
    }

    if (doc && doc.addEventListener) {
        doc.addEventListener('submit', onSubmit, true);
        doc.addEventListener('click', onClick, true);
    }

    /* ------------------------------------------------------------ export */

    global.Insights = {
        __ui: 1,
        toast: toast,
        confirm: confirm,
        busy: busy,
        explain: explain,
        fieldError: fieldError,
        deferred: deferred,
        _flushPending: flushPending
    };

    // The old name, kept because pages and gate scripts call it. A page that
    // still defines its own showToast replaces this one; that copy is debt the
    // ratchet in scripts/check_ux.py counts.
    if (typeof global.showToast !== 'function') {
        global.showToast = function (message, type) { return toast(message, type); };
    }
})(window);
