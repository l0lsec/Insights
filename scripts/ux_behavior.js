// Runs static/insights-ui.js under node, in a small fake DOM, and checks that
// each shared piece does what docs/UX_PRINCIPLES.md promises. Called by
// `python scripts/check_ux.py behavior`, which reads the JSON this prints.
//
//   node scripts/ux_behavior.js <path to insights-ui.js>
'use strict';
const fs = require('fs');
const vm = require('vm');

const SOURCE = fs.readFileSync(process.argv[2], 'utf8');

// ---------------------------------------------------------------- fake DOM
class Node_ {
  constructor(doc) { this.ownerDocument = doc; this.parentNode = null; this.childNodes = []; this.listeners = {}; }
  appendChild(child) { return this.insertBefore(child, null); }
  insertBefore(child, ref) {
    if (child.parentNode) child.parentNode.removeChild(child);
    const at = ref ? this.childNodes.indexOf(ref) : -1;
    if (at === -1) this.childNodes.push(child); else this.childNodes.splice(at, 0, child);
    child.parentNode = this;
    return child;
  }
  removeChild(child) {
    const at = this.childNodes.indexOf(child);
    if (at !== -1) this.childNodes.splice(at, 1);
    child.parentNode = null;
    return child;
  }
  get nextSibling() {
    if (!this.parentNode) return null;
    const list = this.parentNode.childNodes;
    return list[list.indexOf(this) + 1] || null;
  }
  contains(other) {
    for (let n = other; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn); }
  dispatch(type) {
    const event = { type, target: this, preventDefault() {}, stopPropagation() {} };
    (this.listeners[type] || []).slice().forEach(fn => fn.call(this, event));
  }
  get textContent() { return this.childNodes.map(c => c.textContent).join(''); }
  set textContent(value) { this.childNodes = []; if (value !== '') this.appendChild(new Text_(this.ownerDocument, value)); }
}
class Text_ extends Node_ {
  constructor(doc, text) { super(doc); this.data = String(text); this.nodeType = 3; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
  get html() { return this.data.replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
}
class Raw_ extends Text_ { get html() { return this.data; } get textContent() { return this.data.replace(/<[^>]*>/g, ''); } }
class Element_ extends Node_ {
  constructor(doc, tag) { super(doc); this.tagName = tag.toUpperCase(); this.attrs = {}; this.nodeType = 1; this.disabled = false; this.hidden = false; this.value = ''; }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.attrs.id = String(v); }
  get className() { return this.attrs.class || ''; }
  set className(v) { this.attrs.class = String(v); }
  get classList() {
    const self = this;
    const list = () => self.className.split(/\s+/).filter(Boolean);
    return {
      add(...names) { const l = list(); names.forEach(n => { if (!l.includes(n)) l.push(n); }); self.className = l.join(' '); },
      remove(...names) { self.className = list().filter(n => !names.includes(n)).join(' '); },
      contains(name) { return list().includes(name); },
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
  removeAttribute(k) { delete this.attrs[k]; }
  focus() { this.ownerDocument.activeElement = this; }
  click() {
    if (this.disabled) return;
    if (this.ownerDocument.fire) this.ownerDocument.fire('click', this, () => { if (this.onDefault) this.onDefault(); });
    else this.dispatch('click');
  }
  get innerHTML() { return this.childNodes.map(c => c.html).join(''); }
  set innerHTML(v) { this.childNodes = []; if (v !== '') this.appendChild(new Raw_(this.ownerDocument, v)); }
  get html() { return `<${this.tagName.toLowerCase()}>${this.innerHTML}</${this.tagName.toLowerCase()}>`; }
  walk(fn) { this.childNodes.forEach(c => { if (c.nodeType === 1) { fn(c); c.walk(fn); } }); }
  querySelectorAll(sel) {
    const m = /^\.([\w-]+)$/.exec(sel); if (!m) throw new Error('fake DOM: unsupported selector ' + sel);
    const out = []; this.walk(n => { if (n.classList.contains(m[1])) out.push(n); }); return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  matches(sel) {
    return sel.split(',').some(part => {
      const m = /^\s*([a-z]*)(?:\[([\w-]+)\])?\s*$/.exec(part);
      if (!m) throw new Error('fake DOM: unsupported selector ' + part);
      return (!m[1] || this.tagName === m[1].toUpperCase()) && (!m[2] || this.hasAttribute(m[2]));
    });
  }
  closest(sel) { for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (n.matches(sel)) return n; return null; }
}
function makeDocument() {
  const doc = { activeElement: null };
  doc.createElement = tag => new Element_(doc, tag);
  doc.createTextNode = text => new Text_(doc, text);
  doc.documentElement = new Element_(doc, 'html');
  doc.body = new Element_(doc, 'body');
  doc.documentElement.appendChild(doc.body);
  doc.getElementById = id => { let hit = null; doc.documentElement.walk(n => { if (!hit && n.id === id) hit = n; }); return hit; };
  doc.activeElement = doc.body;
  doc.listeners = {};
  doc.addEventListener = (type, fn, capture) => { (doc.listeners[type] = doc.listeners[type] || []).push(fn); };
  doc.querySelector = () => null;
  // A user event: document capture listeners first, then the target's own, then the default action.
  doc.fire = (type, target, defaultAction) => {
    const event = { type, target, defaultPrevented: false, stopped: false,
                    preventDefault() { this.defaultPrevented = true; },
                    stopImmediatePropagation() { this.stopped = true; }, stopPropagation() {} };
    const pending = (doc.listeners[type] || []).map(fn => fn(event));
    if (!event.stopped) (target.listeners[type] || []).slice().forEach(fn => fn.call(target, event));
    if (!event.defaultPrevented && defaultAction) defaultAction();
    return Promise.all(pending.map(p => Promise.resolve(p)));
  };
  return doc;
}

// ------------------------------------------------------------ fake clock
function makeClock() {
  let now = 0, seq = 0; const timers = new Map();
  return {
    setTimeout(fn, ms) { const id = ++seq; timers.set(id, { at: now + (ms || 0), fn }); return id; },
    clearTimeout(id) { timers.delete(id); },
    async tick(ms) {
      const end = now + ms;
      for (;;) {
        await flush();
        let next = null;
        for (const [id, t] of timers) if (t.at <= end && (!next || t.at < next[1].at)) next = [id, t];
        if (!next) break;
        timers.delete(next[0]); now = next[1].at; next[1].fn();
      }
      now = end; await flush();
    },
  };
}
function flush() { return new Promise(r => setImmediate(r)); }

// --------------------------------------------------------------- world
function world(opts = {}) {
  const document = makeDocument();
  const clock = makeClock();
  const region = document.createElement('div');
  region.id = 'insightsToasts'; region.setAttribute('aria-live', 'polite');
  if (!opts.noRegion) document.body.appendChild(region);
  const nativeConfirms = [];
  const sent = [];
  const win = {
    FormData: class { constructor(form) { this.form = form; } },
    fetch: (url, init) => { sent.push({ url, init: init || {} }); return Promise.resolve({ ok: opts.fetchOk !== false, status: opts.fetchOk === false ? 500 : 200 }); },
    location: { reloads: 0, href: '', reload() { this.reloads++; } },
    document, console, Promise, WeakMap, setImmediate,
    navigator: { onLine: true },
    setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout,
    confirm: m => { nativeConfirms.push(m); return opts.nativeAnswer !== false; },
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    dispatch(type) { (this.listeners[type] || []).forEach(fn => fn({ type })); },
  };
  if (opts.showToast) win.showToast = opts.showToast;
  if (!opts.noBootstrap) {
    win.bootstrap = { Modal: { getOrCreateInstance(el) {
      el.__modal = el.__modal || {
        shown: false,
        show() { this.shown = true; Promise.resolve().then(() => el.dispatch('shown.bs.modal')); },
        hide() { if (!this.shown) return; this.shown = false; el.dispatch('hidden.bs.modal'); },
      };
      return el.__modal;
    } } };
  }
  win.window = win;
  vm.createContext(win);
  vm.runInContext(SOURCE, win, { filename: 'insights-ui.js' });
  return { win, document, clock, region, nativeConfirms, sent, I: win.Insights };
}

const results = {};
async function test(name, fn) {
  try { await fn(); results[name] = true; }
  catch (e) { results[name] = String(e && e.message || e); }
}
function ok(cond, msg) { if (!cond) throw new Error(msg); }
const toasts = w => w.region.querySelectorAll('.insights-toast');
const buttonByText = (root, text) => { let hit = null; root.walk(n => { if (!hit && n.tagName === 'BUTTON' && n.textContent === text) hit = n; }); return hit; };

(async () => {
  await test('toast_success_auto_dismisses', async () => {
    const w = world();
    w.I.toast('<b>Saved</b> the post', 'success');
    const [t] = toasts(w);
    ok(t, 'no toast in #insightsToasts');
    ok(t.getAttribute('role') === 'status', 'a success toast is not role=status');
    ok(t.textContent.includes('<b>Saved</b>'), 'the message was not kept as text (HTML would be injected)');
    await w.clock.tick(4900); ok(toasts(w).length === 1, 'gone before 5 s');
    await w.clock.tick(200); ok(toasts(w).length === 0, 'a success toast is still up after 5 s');
  });

  await test('toast_error_stays_until_dismissed', async () => {
    const w = world();
    w.I.toast('Could not save. Try again.', { type: 'error' });
    await w.clock.tick(120000);
    const [t] = toasts(w);
    ok(t, 'an error toast vanished on its own');
    ok(t.getAttribute('role') === 'alert', 'an error toast is not role=alert');
    const close = t.querySelector('.insights-toast-close');
    ok(close && close.getAttribute('aria-label') === 'Dismiss' && close.getAttribute('type') === 'button', 'no labelled Dismiss button');
    close.click();
    ok(toasts(w).length === 0, 'Dismiss did not remove it');
  });

  await test('toast_undo_and_action_fire_once', async () => {
    const w = world(); let undos = 0, retries = 0;
    w.I.toast('Deleted.', { type: 'success', undo: () => undos++ });
    buttonByText(w.region, 'Undo').click();
    ok(undos === 1 && toasts(w).length === 0, 'Undo did not run once and close the toast');
    w.I.toast('Failed.', { type: 'error', action: { label: 'Retry', run: () => retries++ } });
    buttonByText(w.region, 'Retry').click();
    ok(retries === 1 && toasts(w).length === 0, 'Retry did not run once and close the toast');
  });

  await test('toast_pauses_while_hovered', async () => {
    const w = world();
    w.I.toast('Queued.', 'success');
    const [t] = toasts(w);
    t.dispatch('mouseenter'); await w.clock.tick(30000);
    ok(toasts(w).length === 1, 'it vanished while the pointer was on it');
    t.dispatch('mouseleave'); await w.clock.tick(5100);
    ok(toasts(w).length === 0, 'it did not go after the pointer left');
  });

  await test('toast_stack_is_short_and_keeps_errors', async () => {
    const w = world();
    w.I.toast('First failure', 'error');
    for (let i = 0; i < 6; i++) w.I.toast('note ' + i, 'info');
    const shown = toasts(w);
    ok(shown.length <= 4, `${shown.length} toasts stacked`);
    ok(shown.some(t => t.getAttribute('role') === 'alert'), 'an error was pushed out by notes');
  });

  await test('toast_region_created_when_missing', async () => {
    const w = world({ noRegion: true });
    w.I.toast('hello');
    const region = w.document.getElementById('insightsToasts');
    ok(region && region.getAttribute('aria-live') === 'polite' && region.querySelectorAll('.insights-toast').length === 1, 'no live region was made');
  });

  await test('confirm_resolves_and_focuses_safely', async () => {
    const w = world();
    const opener = w.document.createElement('button'); w.document.body.appendChild(opener); opener.focus();
    const danger = w.I.confirm({ title: 'Delete 3 posts?', body: 'They leave every queue.', confirmLabel: 'Delete 3 posts', danger: true });
    await flush();
    const modal = w.document.getElementById('insightsConfirm');
    ok(modal, 'no dialog was built');
    const okBtn = buttonByText(modal, 'Delete 3 posts'), cancel = buttonByText(modal, 'Cancel');
    ok(okBtn && cancel, 'the dialog does not show the labels it was given');
    ok(w.document.activeElement === cancel, 'a risky confirm does not start on Cancel');
    ok(okBtn.classList.contains('btn-danger'), 'a risky confirm is not styled as danger');
    ok(modal.textContent.includes('They leave every queue.'), 'the body is missing');
    cancel.click();
    ok((await danger) === false, 'Cancel did not resolve false');
    ok(w.document.activeElement === opener, 'focus did not go back to what opened it');

    const plain = w.I.confirm('Queue this post?');
    await flush();
    const okPlain = buttonByText(modal, 'Confirm');
    ok(w.document.activeElement === okPlain, 'a safe confirm does not start on its action');
    ok(okPlain.classList.contains('btn-primary') && !okPlain.classList.contains('btn-danger'), 'a safe confirm kept the danger style');
    okPlain.click();
    ok((await plain) === true, 'Confirm did not resolve true');
  });

  await test('confirm_waits_its_turn', async () => {
    const w = world();
    const first = w.I.confirm({ title: 'First?' });
    const second = w.I.confirm({ title: 'Second?' });
    await flush();
    const title = w.document.getElementById('insightsConfirmTitle');
    ok(title.textContent === 'First?', 'the second confirm replaced the first while it was open');
    buttonByText(w.document.getElementById('insightsConfirm'), 'Confirm').click();
    ok((await first) === true, 'first did not resolve');
    await flush(); await flush();
    ok(title.textContent === 'Second?', 'the second confirm never opened');
    buttonByText(w.document.getElementById('insightsConfirm'), 'Cancel').click();
    ok((await second) === false, 'second did not resolve false');
  });

  await test('confirm_without_bootstrap_uses_native', async () => {
    const w = world({ noBootstrap: true, nativeAnswer: false });
    const answer = await w.I.confirm({ title: 'Delete it?', body: 'This cannot be undone.' });
    ok(answer === false && w.nativeConfirms.length === 1 && w.nativeConfirms[0].includes('cannot be undone'), 'no native fallback');
  });

  await test('busy_disables_at_once_and_restores', async () => {
    const w = world();
    const btn = w.document.createElement('button'); btn.innerHTML = '<b>Save</b>';
    let release, starts = 0;
    const work = () => { starts++; return new Promise(r => { release = r; }); };
    const p = w.I.busy(btn, work, { label: 'Saving...' });
    ok(btn.disabled && btn.getAttribute('aria-busy') === 'true', 'the button was not disabled at once');
    ok(btn.textContent.includes('Saving...'), 'the button does not say what it is doing');
    const again = w.I.busy(btn, work);
    await flush();
    ok(again === p && starts === 1, 'a second click started the work again');
    release(42);
    ok((await p) === 42, 'the result was not passed through');
    ok(btn.innerHTML === '<b>Save</b>' && !btn.disabled && !btn.hasAttribute('aria-busy'), 'the button did not come back as it was');

    const failing = w.I.busy(btn, () => Promise.reject(new Error('boom')));
    let caught = null; try { await failing; } catch (e) { caught = e; }
    ok(caught && caught.message === 'boom', 'a failure was swallowed');
    ok(btn.innerHTML === '<b>Save</b>' && !btn.disabled, 'a failure left a dead button');
    const sync = w.I.busy(btn, () => { throw new Error('sync'); });
    caught = null; try { await sync; } catch (e) { caught = e; }
    ok(caught && !btn.disabled, 'a synchronous throw left a dead button');
  });

  await test('explain_says_it_plainly', async () => {
    const w = world(); const x = w.I.explain;
    ok(/connection/.test(x(Object.assign(new Error('Failed to fetch'), { name: 'TypeError' }))), 'a network failure is not explained as one');
    ok(/on its side/.test(x({ status: 502 })), 'a 5xx is not explained');
    ok(/no longer exists/.test(x({ status: 404 })), 'a 404 is not explained');
    ok(!/Unexpected token/.test(x(new Error("Unexpected token '<', \"<!doctype \"... is not valid JSON"))), 'a raw parser error reached the user');
    ok(x(new Error('Connect LinkedIn to post there.')) === 'Connect LinkedIn to post there.', 'a plain message was not passed through');
    ok(x({ error: 'Pick at least one platform.', status: 400 }) === 'Pick at least one platform.', "the server's own message was dropped");
    ok(x({ status: 500 }, 'Could not save the post.').startsWith('Could not save the post. '), 'the context was not put first');
    ok(x(Object.assign(new Error('x is not a function'), { userMessage: 'Reload the page.' })) === 'Reload the page.', 'userMessage was not honored');
    ok(!/undefined|null/.test(x(undefined)) && !/undefined/.test(x('Cannot read properties of undefined')), 'an empty or technical error reached the user');
  });

  await test('field_error_keeps_input_and_clears', async () => {
    const w = world();
    const holder = w.document.createElement('div'); w.document.body.appendChild(holder);
    const input = w.document.createElement('input'); input.id = 'postTime'; input.value = '25:99'; holder.appendChild(input);
    const after = w.document.createElement('span'); holder.appendChild(after);
    w.I.fieldError(input, 'Enter a time like 09:30.', { focus: true });
    const note = w.document.getElementById('postTime-error');
    ok(note && note.textContent === 'Enter a time like 09:30.' && input.nextSibling === note, 'the message is not next to the field');
    ok(input.getAttribute('aria-invalid') === 'true' && input.classList.contains('is-invalid'), 'the field is not marked invalid');
    ok((input.getAttribute('aria-describedby') || '').split(' ').includes('postTime-error'), 'the message is not tied to the field');
    ok(input.value === '25:99', "the user's input was changed");
    ok(w.document.activeElement === input, 'focus did not go to the field');
    w.I.fieldError(input, 'Still wrong.');
    ok(holder.querySelectorAll('.invalid-feedback').length === 1, 'a second error stacked a second note');
    input.dispatch('input');
    ok(!w.document.getElementById('postTime-error') && !input.hasAttribute('aria-invalid') && !input.classList.contains('is-invalid'), 'editing did not clear it');
  });

  await test('deferred_undo_sends_nothing', async () => {
    const w = world(); const log = [];
    const p = w.I.deferred(() => { log.push('run'); return Promise.resolve({ ok: true }); }, {
      message: 'Deleted 3 posts.', hide: () => log.push('hide'), restore: () => log.push('restore') });
    ok(log.join() === 'hide', 'the item was not hidden at once');
    const [t] = toasts(w);
    ok(t && t.textContent.includes('Deleted 3 posts.'), 'no toast says what happened');
    buttonByText(w.region, 'Undo').click();
    ok((await p).undone === true, 'Undo did not report undone');
    await w.clock.tick(60000);
    ok(log.join() === 'hide,restore', `Undo still sent the request (${log.join()})`);
  });

  await test('deferred_commits_after_the_window', async () => {
    const w = world(); const seen = [];
    const p = w.I.deferred(o => { seen.push(o.keepalive); return Promise.resolve({ ok: true, status: 200 }); }, { message: 'Deleted.' });
    await w.clock.tick(5900);
    ok(seen.length === 0, 'it was sent before the Undo window closed');
    await w.clock.tick(200);
    const r = await p;
    ok(seen.length === 1 && seen[0] === false && r.committed, 'it was not sent once after the window');
    ok(toasts(w).length === 0, 'the Undo toast outlived the request');
  });

  await test('deferred_failure_restores_and_offers_retry', async () => {
    const w = world(); const log = []; let calls = 0;
    const p = w.I.deferred(() => { calls++; return Promise.resolve(calls === 1 ? { ok: false, status: 500 } : { ok: true }); }, {
      message: 'Deleted.', hide: () => log.push('hide'), restore: () => log.push('restore') });
    await w.clock.tick(6000);
    const r = await p;
    ok(r.failed && log.join() === 'hide,restore', 'a failed request did not put the item back');
    const [t] = toasts(w);
    ok(t && t.getAttribute('role') === 'alert' && /on its side/.test(t.textContent), 'the failure was not explained in an error toast');
    buttonByText(w.region, 'Retry').click();
    await flush(); await flush();
    ok(calls === 2 && log.join() === 'hide,restore,hide', 'Retry did not hide and send again');
  });

  await test('deferred_flushes_on_pagehide', async () => {
    const w = world(); const seen = [];
    const p = w.I.deferred(o => { seen.push(o.keepalive); return Promise.resolve({ ok: true }); }, { message: 'Deleted.' });
    w.win.dispatch('pagehide');
    const r = await p;
    ok(seen.length === 1 && seen[0] === true && r.committed, 'leaving the page did not send it with keepalive');
    await w.clock.tick(10000);
    ok(seen.length === 1, 'it was sent twice');
  });

  await test('loaded_once_and_show_toast_kept', async () => {
    const w = world();
    const first = w.win.Insights;
    vm.runInContext(SOURCE, w.win);
    ok(w.win.Insights === first, 'loading it twice replaced the first copy');
    w.win.showToast('legacy call', 'success');
    ok(toasts(w).length === 1 && toasts(w)[0].classList.contains('insights-toast-success'), 'showToast(message, type) does not reach Insights.toast');
    let own = 0;
    const w2 = world({ showToast: () => own++ });
    w2.win.showToast('x');
    ok(own === 1, "a page's own showToast was overwritten");
  });

  // ------------------------------------------------ declarative markup
  const confirmButton = w => buttonByText(w.document.body, 'Delete feed');
  const cancelButton = w => buttonByText(w.document.body, 'Cancel');
  function feedRow(w, attrs) {
    const tr = w.document.createElement('tr');
    const form = w.document.createElement('form');
    form.action = '/feed/3/delete'; form.setAttribute('method', 'post');
    Object.entries(attrs).forEach(([k, v]) => form.setAttribute(k, v));
    form.submits = 0;
    form.requestSubmit = () => { form.submits++; };
    tr.appendChild(form); w.document.body.appendChild(tr);
    return { tr, form };
  }

  await test('markup_confirm_form_waits_for_the_answer', async () => {
    let w = world();
    let { form } = feedRow(w, { 'data-confirm': 'Its 12 episodes go too.', 'data-confirm-title': 'Delete feed?', 'data-confirm-label': 'Delete feed', 'data-confirm-danger': '' });
    let fired = w.document.fire('submit', form, () => form.submits++);
    await flush();
    ok(form.submits === 0, 'the form submitted before the question was answered');
    ok(w.document.activeElement === cancelButton(w), 'a danger confirm from markup did not start on Cancel');
    cancelButton(w).click(); await fired; await flush();
    ok(form.submits === 0, 'Cancel still submitted the form');
    fired = w.document.fire('submit', form, () => form.submits++);
    await flush();
    confirmButton(w).click(); await fired; await flush();
    ok(form.submits === 1, `Confirm did not submit exactly once (${form.submits})`);
  });

  await test('markup_undo_form_hides_then_sends_after_the_window', async () => {
    const w = world();
    const { tr, form } = feedRow(w, { 'data-undo': 'Deleted the feed.', 'data-undo-row': 'tr', 'data-undo-then': 'reload' });
    const done = w.document.fire('submit', form, () => form.submits++);
    await flush();
    ok(tr.hidden === true && w.sent.length === 0 && form.submits === 0, 'the row did not hide at once, or it was sent without waiting');
    ok(/Deleted the feed/.test(toasts(w)[0].textContent) && buttonByText(w.region, 'Undo'), 'no Undo toast naming what was deleted');
    await w.clock.tick(6100); await done; await flush();
    ok(w.sent.length === 1 && w.sent[0].url === '/feed/3/delete' && w.sent[0].init.method === 'POST', 'it was not sent once, as a POST, after the window');
    ok(w.win.location.reloads === 1, 'data-undo-then="reload" did not reload after it went through');
  });

  await test('markup_undo_form_undo_sends_nothing', async () => {
    const w = world();
    const { tr, form } = feedRow(w, { 'data-undo': 'Deleted the feed.', 'data-undo-row': 'tr' });
    w.document.fire('submit', form);
    await flush();
    buttonByText(w.region, 'Undo').click();
    await w.clock.tick(60000);
    ok(tr.hidden === false && w.sent.length === 0, 'Undo did not bring the row back, or it was still sent');
  });

  await test('markup_confirm_button_runs_its_handler_only_on_yes', async () => {
    const w = world(); let ran = 0;
    const b = w.document.createElement('button');
    b.setAttribute('data-confirm', 'Disconnect LinkedIn?'); b.setAttribute('data-confirm-label', 'Delete feed');
    b.addEventListener('click', () => ran++);
    w.document.body.appendChild(b);
    let fired = w.document.fire('click', b); await flush();
    ok(ran === 0, 'the button handler ran before the question was answered');
    cancelButton(w).click(); await fired; await flush();
    ok(ran === 0, 'Cancel still ran the handler');
    fired = w.document.fire('click', b); await flush();
    confirmButton(w).click(); await fired; await flush();
    ok(ran === 1, `Confirm did not run the handler exactly once (${ran})`);
  });

  process.stdout.write(JSON.stringify(results) + '\n');
})();
