"""The one stand-in for window.Insights that gate harnesses put in their node world.

Gate scripts pull a page's inline <script> out of the rendered HTML and run it in
a vm context. They skip <script src=...>, so static/insights-ui.js never loads
there, and a page that calls Insights.toast/confirm/... would throw. Every
harness that stubs alert/confirm/showToast therefore also puts this in its
world, writing into the SAME log it already asserts on:

    from _insights_stub import INSIGHTS_STUB_JS
    JS_HARNESS = INSIGHTS_STUB_JS + r'''
      ...
      ctx.Insights = makeInsights(ctx);
      vm.createContext(ctx);
    '''

makeInsights(stubs) routes every call into the harness's OWN stubs, so the log,
the answers and the assertions stay exactly as they were:

    const ctx = { alert: ..., confirm: ..., showToast: ..., ... };
    ctx.Insights = makeInsights(ctx);          // or makeInsights({alert, confirm, showToast})

    toast(m, type)    -> stubs.alert(m) when the harness stubs alert (it replaced alert),
                         else stubs.showToast(m, type)
    confirm(o)        -> Promise.resolve(stubs.confirm(title + '\\n\\n' + body)); a harness
                         that wants the options sees them as stubs.confirmOpts(o)
    deferred(run, o)  -> stubs.deferred(o) if given; hide(); then, unless stubs.undo is
                         true, run({keepalive: false}) at once (the Undo window is the
                         browser's business; the behaviour gate covers it). A failed run
                         restores and is reported through toast(..., 'error')
    done(m, next)     -> stubs.done(m, next) if given, then toast(m, 'success')
    busy(btn, work)   -> runs work
    fieldError(i, m)  -> stubs.fieldError(i, m) if given, else toast(m, 'error')
    explain           -> the real one, read from static/insights-ui.js, so a harness
                         asserts the copy users actually see

`python scripts/check_ux.py harness` fails if a harness stubs alert/confirm/showToast
without this, or has fewer check(...) lines than on main.
"""
import json
import os

UI_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "insights-ui.js")

INSIGHTS_STUB_JS = r"""
const makeInsights = (() => {
  const realUi = (() => {
    const fsInsights = require('fs');
    const vmInsights = require('vm');
    const win = { navigator: { onLine: true }, setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout, Promise: globalThis.Promise, WeakMap: globalThis.WeakMap };
    win.window = win;
    vmInsights.createContext(win);
    vmInsights.runInContext(fsInsights.readFileSync(__UI_JS__, 'utf8'), win, { filename: 'insights-ui.js' });
    return win.Insights;
  })();
  return function makeInsights(stubs) {
    stubs = stubs || {};
    const text = (o) => typeof o === 'string' ? o : [o && o.title, o && o.body].filter(Boolean).join('\n\n');
    const toast = (message, typeOrOpts) => {
      const type = typeof typeOrOpts === 'string' ? typeOrOpts : (typeOrOpts && typeOrOpts.type) || 'info';
      if (typeof stubs.alert === 'function') stubs.alert(message);
      else if (typeof stubs.showToast === 'function') stubs.showToast(message, type);
      return { dismiss() {} };
    };
    return {
      __ui: 1,
      toast,
      done(message, next) {
        if (typeof stubs.done === 'function') stubs.done(message, next);
        return toast(message, 'success');
      },
      confirm(o) {
        if (typeof stubs.confirmOpts === 'function') stubs.confirmOpts(o);
        return Promise.resolve(typeof stubs.confirm === 'function' ? stubs.confirm(text(o)) : true);
      },
      async busy(btn, work) { return typeof work === 'function' ? work() : work; },
      explain: (err, context) => realUi.explain(err, context),
      fieldError(input, message) {
        if (typeof stubs.fieldError === 'function') stubs.fieldError(input, message);
        else if (message) toast(message, 'error');
        return input;
      },
      async deferred(run, o) {
        o = o || {};
        if (typeof stubs.deferred === 'function') stubs.deferred(o);
        if (o.hide) o.hide();
        if (stubs.undo === true) { if (o.restore) o.restore(); return { undone: true }; }
        try {
          const value = await run({ keepalive: false });
          if (value && typeof value.ok === 'boolean' && !value.ok) throw value;
          return { committed: true, value };
        } catch (error) {
          if (o.restore) o.restore();
          toast(realUi.explain(error, o.failMessage || 'That did not go through, so it is back.'), 'error');
          return { failed: true, error };
        }
      },
    };
  };
})();
""".replace("__UI_JS__", json.dumps(UI_JS))
