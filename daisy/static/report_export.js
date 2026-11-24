(function(){
  var DATA = JSON.parse(document.getElementById('daisy-data').textContent);
  function $(id){ return document.getElementById(id); }
  var sectionsEl=$('sections'), tabsEl=$('tabs'), controlsEl=$('controls'), selectEl=$('feature'),
      prevEl=$('prev'), nextEl=$('next'), plotsEl=$('plots'), emptyEl=$('empty'), noteEl=$('note'),
      staticEl=$('static'), bandEl=$('badge'), posEl=$('position');
  var cache = {};
  var viewsEl = $('views'), viewsWrap = $('views-wrap');
  var state = {section: 0, view: DATA.defaultView || DATA.views[0], group: DATA.sections.map(function(){ return 0; }), index: {}};
  var CONFIG = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']};

  function section(){ return DATA.sections[state.section]; }
  function groups(){ return section().groups || []; }
  function group(){ return groups()[state.group[state.section]] || null; }
  function key(){ return state.section + ':' + state.group[state.section]; }
  function viewOf(f){
    if (state.view === DATA.views[0]) { return f; }
    return (f.views && f.views[state.view]) || {error: 'Nothing to show for ' + state.view + ' rows.', figs: []};
  }
  function figure(f, v, i){
    var k = f.id + ':' + state.view + ':' + i;
    if (!cache[k]) { cache[k] = JSON.parse(v.figs[i].fig); }
    return cache[k];
  }
  function drawViews(){
    viewsEl.innerHTML = '';
    show(viewsWrap, DATA.views.length > 1);
    DATA.views.forEach(function(name){
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = name;
      b.setAttribute('aria-pressed', name === state.view ? 'true' : 'false');
      b.addEventListener('click', function(){ state.view = name; drawViews(); drawSection(); });
      viewsEl.appendChild(b);
    });
  }
  function show(el, on){ el.style.display = on ? '' : 'none'; }
  function purge(){
    Array.prototype.slice.call(plotsEl.children).forEach(function(c){ try { Plotly.purge(c); } catch (e) {} });
    plotsEl.innerHTML = '';
  }

  function drawSections(){
    sectionsEl.innerHTML = '';
    DATA.sections.forEach(function(s, i){
      var b = document.createElement('button');
      b.className = 'section-btn'; b.type = 'button'; b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', i === state.section ? 'true' : 'false');
      b.textContent = s.title;
      b.addEventListener('click', function(){ state.section = i; drawSection(); });
      sectionsEl.appendChild(b);
    });
  }
  function drawSection(){
    drawSections();
    var s = section();
    noteEl.textContent = s.note || ''; show(noteEl, !!s.note);
    if (s.kind === 'html') {
      purge(); show(tabsEl, false); show(controlsEl, false); show(plotsEl, false); show(emptyEl, false);
      staticEl.innerHTML = (s.views && s.views[state.view] !== undefined && state.view !== DATA.views[0]) ? s.views[state.view] : s.html;
      show(staticEl, true);
      return;
    }
    staticEl.innerHTML = ''; show(staticEl, false); show(plotsEl, true);
    drawTabs(); drawGroup();
  }
  function drawTabs(){
    tabsEl.innerHTML = '';
    var gs = groups();
    tabsEl.hidden = gs.length < 2; show(tabsEl, gs.length > 1);
    gs.forEach(function(g, i){
      var b = document.createElement('button');
      b.className = 'tab'; b.type = 'button'; b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', i === state.group[state.section] ? 'true' : 'false');
      b.appendChild(document.createTextNode(g.title));
      var c = document.createElement('span'); c.className = 'count'; c.textContent = g.features.length;
      b.appendChild(c);
      b.addEventListener('click', function(){ state.group[state.section] = i; drawTabs(); drawGroup(); });
      tabsEl.appendChild(b);
    });
  }
  function drawGroup(){
    var g = group(), has = !!g && g.features.length > 0;
    selectEl.innerHTML = '';
    if (has) { g.features.forEach(function(f, i){
      var o = document.createElement('option'); o.value = i; o.textContent = f.name; selectEl.appendChild(o);
    }); }
    selectEl.disabled = !has;
    show(controlsEl, has && !section().single); show(emptyEl, !has);
    emptyEl.textContent = (g && g.empty) || 'No features in this group.';
    if (has) { selectEl.value = state.index[key()] || 0; }
    drawPlot();
  }
  function drawPlot(){
    purge();
    var g = group(), n = g ? g.features.length : 0;
    prevEl.disabled = nextEl.disabled = n < 2;
    if (!n) { return; }
    var i = state.index[key()] || 0, f = g.features[i], v = viewOf(f);
    bandEl.textContent = v.badge || ''; show(bandEl, !!v.badge);
    posEl.textContent = (i + 1) + ' of ' + n;
    if (v.error) {
      var err = document.createElement('div'); err.className = 'error'; err.textContent = v.error;
      plotsEl.appendChild(err); return;
    }
    v.figs.forEach(function(item, j){
      if (item.title) {
        var h = document.createElement('h3'); h.className = 'plot-title'; h.textContent = item.title;
        plotsEl.appendChild(h);
      }
      var div = document.createElement('div'); div.className = 'plot';
      var fig = figure(f, v, j), showErrors = !!$('errbars') && $('errbars').checked;
      (fig.data || []).forEach(function(t){ if (t.error_y) { t.error_y.visible = showErrors; } });
      div.style.height = ((fig.layout && fig.layout.height) || 480) + 'px';
      plotsEl.appendChild(div);
      Plotly.react(div, fig.data, fig.layout, CONFIG);
    });
    if (v.html) {
      var table = document.createElement('div'); table.style.marginTop = '18px'; table.innerHTML = v.html;
      plotsEl.appendChild(table);
    }
  }
  function move(step){
    var n = group().features.length, k = key();
    state.index[k] = ((state.index[k] || 0) + step + n) % n;
    selectEl.value = state.index[k]; drawPlot();
  }
  selectEl.addEventListener('change', function(){ state.index[key()] = +selectEl.value; drawPlot(); });
  if ($('errbars')) { $('errbars').checked = !!DATA.errorBars; $('errbars').addEventListener('change', drawPlot); }
  prevEl.addEventListener('click', function(){ move(-1); });
  nextEl.addEventListener('click', function(){ move(1); });
  document.addEventListener('keydown', function(e){
    var g = group();
    if (e.target === selectEl || e.altKey || e.ctrlKey || e.metaKey || section().kind === 'html' || !g || !g.features.length) { return; }
    if (e.key === 'ArrowLeft') { move(-1); } else if (e.key === 'ArrowRight') { move(1); }
  });
  drawViews();
  drawSection();
})();
