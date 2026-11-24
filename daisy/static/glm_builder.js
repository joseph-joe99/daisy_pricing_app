(function () {
  if (window.__daisyGlmBuilder) { return; }
  window.__daisyGlmBuilder = true;

  var TYPES = {
    numeric: [
      ["linear", "Linear"],
      ["bs", "Spline: B-spline bs()"],
      ["cr", "Spline: natural cubic cr()"],
      ["cc", "Spline: cyclic cc()"],
      ["poly", "Polynomial"],
      ["log", "Log"],
      ["sqrt", "Square root"],
      ["square", "Square"],
      ["clip", "Clip to min / max"],
      ["center", "Centred center()"]
    ],
    categorical: [["cat", "Categorical C()"]]
  };
  var OPS = [[":", "Interaction only  a:b"], ["*", "Crossed  a*b (includes both)"], ["/", "Nested  a/b (a + a:b)"]];

  var terms = [];
  var nextId = 1;
  var intercept = true;

  function $(id) { return document.getElementById(id); }

  function h(tag, props, kids) {
    var e = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      if (k === "text") { e.textContent = props[k]; }
      else if (k.indexOf("on") === 0) { e.addEventListener(k.slice(2), props[k]); }
      else if (k === "value") { e.value = props[k]; }
      else { e.setAttribute(k, props[k]); }
    });
    (kids || []).forEach(function (c) { if (c) { e.appendChild(c); } });
    return e;
  }

  // A value as a Python literal, for the formula text.
  function pyLit(v) {
    if (v === true) { return "True"; }
    if (v === false) { return "False"; }
    if (v === null || v === undefined) { return "None"; }
    if (typeof v === "number") { return String(v); }
    return JSON.stringify(String(v));
  }

  function ident(name) {
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : 'Q("' + String(name).replace(/"/g, "") + '")';
  }

  function num(v) {
    var n = parseFloat(v);
    return isFinite(n) ? n : null;
  }

  function df(t) { return Math.max(1, Math.round(t.df || 4)); }

  // The Patsy text for one term.
  function termExpr(t) {
    var x = ident(t.feature);
    switch (t.type) {
      case "cat": return catExpr(t, x);
      case "bs": return "bs(" + x + ", df=" + df(t) + ")";
      case "cr": return "cr(" + x + ", df=" + df(t) + ")";
      case "cc": return "cc(" + x + ", df=" + df(t) + ")";
      case "poly":
        var parts = [x];
        for (var k = 2; k <= Math.max(2, Math.round(t.degree || 2)); k++) { parts.push("I(" + x + " ** " + k + ")"); }
        return parts.join(" + ");
      case "log": return "np.log(" + x + ")";
      case "sqrt": return "np.sqrt(" + x + ")";
      case "square": return "I(" + x + " ** 2)";
      case "center": return "center(" + x + ")";
      case "clip":
        var lo = num(t.lo), hi = num(t.hi);
        if (lo !== null && hi !== null) { return "np.clip(" + x + ", " + lo + ", " + hi + ")"; }
        if (lo !== null) { return "np.maximum(" + x + ", " + lo + ")"; }
        if (hi !== null) { return "np.minimum(" + x + ", " + hi + ")"; }
        return x;
      default: return x;
    }
  }

  // A categorical term with selected levels: each selected level gets its own coefficient, and every other level is
  // grouped as "(other)". The first selected level (in list order) is the baseline. With no levels selected, the term
  // is the plain categorical C(x).
  function catExpr(t, x) {
    var lv = t.levels || [];
    var sel = (t.sel || []).map(function (i) { return lv[i]; }).filter(function (v) { return v !== undefined; });
    if (!sel.length) { return "C(" + x + ")"; }
    return "C(np.where(np.isin(" + x + ", [" + sel.map(pyLit).join(", ") + "]), " + x + ', "(other)"), Treatment(' +
      pyLit(sel[0]) + "))";
  }

  function find(id) {
    for (var i = 0; i < terms.length; i++) {
      if (terms[i].kind === "term" && terms[i].id === Number(id)) { return terms[i]; }
    }
    return null;
  }

  function interactionExpr(t) {
    var a = find(t.a), b = find(t.b);
    if (!a || !b) { return null; }
    return "(" + termExpr(a) + ") " + t.op + " (" + termExpr(b) + ")";
  }

  function rowExpr(t) { return t.kind === "term" ? termExpr(t) : interactionExpr(t); }

  function formula() {
    var parts = [];
    terms.forEach(function (t) {
      var e = rowExpr(t);
      if (e) { parts.push(e); }
    });
    if (!parts.length) { return ""; }
    return (intercept ? "" : "0 + ") + parts.join(" + ");
  }

  // Write the formula into the Formula tab's text box, telling Shiny about the change.
  function sync() {
    var f = formula();
    var preview = $("glm-preview");
    if (preview) { preview.textContent = f || "Add features to start the formula."; }
    var box = $("glm_formula");
    if (box && box.value !== f) {
      box.value = f;
      box.dispatchEvent(new Event("input", { bubbles: true }));
      box.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function indexOfId(id) {
    for (var i = 0; i < terms.length; i++) { if (terms[i].id === Number(id)) { return i; } }
    return -1;
  }

  function addTerm(feature, dtype, index, levels) {
    var t = {
      id: nextId++, kind: "term", feature: feature, dtype: dtype,
      type: dtype === "categorical" ? "cat" : "linear", df: 4, degree: 2, lo: "", hi: "",
      levels: levels || [], sel: []
    };
    if (index === undefined || index === null || index < 0 || index > terms.length) { terms.push(t); }
    else { terms.splice(index, 0, t); }
    changed();
  }

  function removeAt(i) {
    var gone = terms[i];
    terms.splice(i, 1);
    terms = terms.filter(function (t) {
      return !(t.kind === "interaction" && (t.a === gone.id || t.b === gone.id));
    });
    changed();
  }

  function moveTo(i, j) {
    if (j < 0 || j >= terms.length) { return; }
    var t = terms.splice(i, 1)[0];
    terms.splice(j, 0, t);
    changed();
  }

  function reorder(id, toIndex) {
    var from = indexOfId(id);
    if (from < 0) { return; }
    var t = terms.splice(from, 1)[0];
    if (from < toIndex) { toIndex -= 1; }
    terms.splice(toIndex, 0, t);
    changed();
  }

  function addInteraction() {
    var a = $("glm-pick-a"), b = $("glm-pick-b"), op = $("glm-op");
    if (!a || !b || !a.value || !b.value) { return; }
    terms.push({ id: nextId++, kind: "interaction", a: Number(a.value), b: Number(b.value), op: op.value });
    changed();
  }

  function termLabel(t) { return t.feature + " (" + t.type + ")"; }

  function refreshPickers() {
    ["glm-pick-a", "glm-pick-b"].forEach(function (id) {
      var s = $(id);
      if (!s) { return; }
      var keep = s.value;
      s.innerHTML = "";
      terms.forEach(function (t) {
        if (t.kind === "term") { s.appendChild(h("option", { value: String(t.id), text: termLabel(t) })); }
      });
      if (keep) { s.value = keep; }
    });
  }

  function numberInput(title, value, min, onChange) {
    return h("label", { class: "d-flex align-items-center gap-1 mb-0" }, [
      h("span", { text: title }),
      h("input", {
        type: "number", class: "form-control form-control-sm", min: String(min), step: "1",
        value: value === "" || value === null || value === undefined ? "" : String(value),
        placeholder: title === "min" || title === "max" ? "none" : "",
        onchange: function () { onChange(this.value); changed(); }
      })
    ]);
  }

  function rowFrame(t, i, content) {
    var up = h("button", { type: "button", class: "btn btn-sm btn-outline-secondary", title: "Move up", text: "\u2191",
      onclick: function () { moveTo(i, i - 1); } });
    var down = h("button", { type: "button", class: "btn btn-sm btn-outline-secondary", title: "Move down", text: "\u2193",
      onclick: function () { moveTo(i, i + 1); } });
    var del = h("button", { type: "button", class: "btn btn-sm btn-outline-danger", title: "Remove", text: "\u2715",
      onclick: function () { removeAt(i); } });
    return h("div", { class: "glm-term" + (t.kind === "interaction" ? " glm-interaction" : ""),
      draggable: "true", "data-term-id": String(t.id) },
      content.concat([h("span", { class: "glm-term-buttons" }, [up, down, del])]));
  }

  function termRow(t, i) {
    var types = TYPES[t.dtype] || TYPES.numeric;
    var select = h("select", {
      class: "form-select form-select-sm", style: "width:auto",
      onchange: function () { t.type = this.value; changed(); }
    }, types.map(function (o) {
      var opt = h("option", { value: o[0], text: o[1] });
      if (o[0] === t.type) { opt.setAttribute("selected", "selected"); }
      return opt;
    }));
    var extra = [];
    if (t.type === "bs" || t.type === "cr" || t.type === "cc") {
      extra.push(numberInput("df", t.df, 1, function (v) { t.df = num(v) || 4; }));
    }
    if (t.type === "poly") {
      extra.push(numberInput("degree", t.degree, 2, function (v) { t.degree = num(v) || 2; }));
    }
    if (t.type === "clip") {
      extra.push(numberInput("min", t.lo, "", function (v) { t.lo = v; }));
      extra.push(numberInput("max", t.hi, "", function (v) { t.hi = v; }));
    }
    if (t.type === "cat" && t.levels && t.levels.length) {
      var selSelect = h("select", {
        multiple: "multiple", size: String(Math.min(4, t.levels.length)), class: "form-select form-select-sm",
        style: "width:auto", title: "Selected levels (blank: every level)",
        onchange: function () {
          t.sel = Array.prototype.map.call(this.selectedOptions, function (o) { return Number(o.value); });
          changed();
        }
      }, t.levels.map(function (v, k) {
        var opt = h("option", { value: String(k), text: String(v) });
        if (t.sel.indexOf(k) >= 0) { opt.setAttribute("selected", "selected"); }
        return opt;
      }));
      extra.push(selSelect);
    }
    var code = h("code", { text: termExpr(t) });
    return rowFrame(t, i, [
      h("span", { class: "glm-term-name", text: t.feature }), select
    ].concat(extra, [code]));
  }

  function interactionRow(t, i) {
    var a = find(t.a), b = find(t.b);
    var label = (a ? a.feature : "?") + " " + t.op + " " + (b ? b.feature : "?");
    return rowFrame(t, i, [
      h("span", { class: "glm-term-name", text: "Interaction " + label }),
      h("code", { text: interactionExpr(t) || "(a term was removed)" })
    ]);
  }

  function renderTerms() {
    var box = $("glm-terms");
    if (!box) { return; }
    var hint = $("glm-hint");
    box.innerHTML = "";
    if (hint) { box.appendChild(hint); hint.style.display = terms.length ? "none" : ""; }
    terms.forEach(function (t, i) {
      box.appendChild(t.kind === "term" ? termRow(t, i) : interactionRow(t, i));
    });
    refreshPickers();
  }

  // Redraw the list and write the formula. Every change goes through here.
  function changed() {
    renderTerms();
    sync();
  }

  function readLevels(chip) {
    try { return JSON.parse(chip.getAttribute("data-levels") || "[]"); } catch (err) { return []; }
  }

  function readPayload(e) {
    try { return JSON.parse(e.dataTransfer.getData("text/plain")); } catch (err) { return null; }
  }

  function onDragStart(e) {
    var target = e.target;
    if (!target || !target.closest) { return; }
    var chip = target.closest(".glm-chip");
    var row = target.closest(".glm-term");
    if (chip) {
      e.dataTransfer.setData("text/plain", JSON.stringify({
        chip: {
          feature: chip.getAttribute("data-feature"), dtype: chip.getAttribute("data-kind"),
          levels: readLevels(chip)
        }
      }));
      e.dataTransfer.effectAllowed = "copy";
    } else if (row && row.hasAttribute("data-term-id")) {
      e.dataTransfer.setData("text/plain", JSON.stringify({ term: Number(row.getAttribute("data-term-id")) }));
      e.dataTransfer.effectAllowed = "move";
    }
  }

  function onDrop(e) {
    var payload = readPayload(e);
    if (!payload) { return; }
    e.preventDefault();
    var row = e.target && e.target.closest ? e.target.closest(".glm-term") : null;
    var toIndex = row ? indexOfId(row.getAttribute("data-term-id")) : terms.length;
    if (toIndex < 0) { toIndex = terms.length; }
    if (payload.term !== undefined) { reorder(payload.term, toIndex); }
    else if (payload.chip) { addTerm(payload.chip.feature, payload.chip.dtype, toIndex, payload.chip.levels); }
  }

  function onClick(e) {
    var chip = e.target && e.target.closest ? e.target.closest(".glm-chip") : null;
    if (chip) { addTerm(chip.getAttribute("data-feature"), chip.getAttribute("data-kind"), null, readLevels(chip)); }
  }

  function init() {
    var box = $("glm-terms");
    if (!box) { return false; }
    if (box.getAttribute("data-ready")) { return true; }
    box.setAttribute("data-ready", "1");

    var ops = $("glm-op");
    if (ops) {
      OPS.forEach(function (o) { ops.appendChild(h("option", { value: o[0], text: o[1] })); });
    }
    document.addEventListener("dragstart", onDragStart);
    document.addEventListener("click", onClick);
    box.addEventListener("dragover", function (e) { e.preventDefault(); });
    box.addEventListener("drop", onDrop);
    var add = $("glm-add-interaction");
    if (add) { add.addEventListener("click", addInteraction); }
    var clear = $("glm-clear");
    if (clear) { clear.addEventListener("click", function () { terms = []; changed(); }); }
    var icpt = $("glm-intercept");
    if (icpt) { icpt.addEventListener("change", function () { intercept = this.checked; sync(); }); }
    renderTerms();
    return true;
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
  window.addEventListener("load", init);
})();
