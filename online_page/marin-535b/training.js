(() => {
  'use strict';
  const events = [{"step":0,"title":"初始 4K 预训练","kind":"recipe","detail":"535.3B / 22.76B active；pooled-wave EP；初始 Harrier 混合。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-5335872267"},{"step":58014,"title":"kernel / router 阶段","kind":"system","detail":"lineage 切入 hero-wd-gate-router-p02-step58k；不能把初始优化器细节当作整条 lineage 的最终配置。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L61"},{"step":81716,"title":"近乎不丢 token 的 EP","kind":"system","detail":"pooled-wave 迁移至 ragged all-to-all；本次运行的系统实现开始不同于首发配置。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-6670507901"},{"step":108000,"title":"切换 main 数据混合","kind":"data","detail":"从永久 checkpoint 恢复后，使用 mixture 996f489106c7b922 的 main 权重；约 27.7% 进度。","source":"https://github.com/marin-community/marin/pull/9162"},{"step":108778,"title":"kernel 优化","kind":"system","detail":"公开汇总的配对测试 MFU 23.6%；这只是该切换点对应的测量。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L94"},{"step":121638,"title":"kernel 优化","kind":"system","detail":"公开汇总的配对测试 MFU 24.1%。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L94"},{"step":146139,"title":"FA4 SM100 阶段","kind":"system","detail":"公开汇总的配对测试 MFU 26.8%；运行名 hero-fa4sm100-nomask-step146k。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L61"},{"step":231464,"title":"kernel 优化","kind":"system","detail":"公开汇总的配对测试 MFU 29.4%；运行名 hero-mfu30-step231k。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L94"},{"step":257553,"title":"4K → 16K","kind":"context","detail":"正式续训使用 16K context；调整 batch，保持每步 46,137,344 tokens。同期 qk_mult 采用 1.48。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-6336070902"},{"step":312192,"title":"切换 cooldown 数据混合","kind":"data","detail":"约 80% 进度，转入选定的 cooldown 权重；提高自然科学和 Q4 的份额。","source":"https://github.com/marin-community/marin/pull/9162"},{"step":356531,"title":"gradclip 续训阶段","kind":"recipe","detail":"lineage 出现 hero-gradclip-step356k。只能确认已存在裁剪相关续训阶段，此报告不猜测未核实阈值。","source":"https://github.com/marin-community/marin/blob/dab2e6d920808f898f565be6dbf39ad01dbdad12/experiments/grug/moe_hero_ep/plot_hero_summary.py#L61"},{"step":372000,"title":"16K → 65K","kind":"context","detail":"最后一段主预训练使用 65,536 context；继续保持每步 token 总数。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-6670507901"},{"step":390251,"title":"18T 主预训练完成","kind":"result","detail":"390,251 次更新；4K 复评 Paloma 1.982，65K 线上末次约 1.960。随后另有 checkpoint 合并及 262K 扩展。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-6670507901"},{"step":393491,"title":"262K 扩展末 checkpoint","kind":"context","detail":"属于主预训练完成后的扩展分支；其评估、导出限制应查看长上下文章节。","source":"https://github.com/marin-community/marin/issues/8435#issuecomment-6748198431"}];
  const select = document.getElementById('train-event-select');
  const detail = document.getElementById('train-event-detail');
  const source = document.getElementById('train-event-source');
  const previous = document.getElementById('train-event-prev');
  const next = document.getElementById('train-event-next');
  if (select && detail && source && previous && next) {
    select.replaceChildren(...events.map((event, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = event.step.toLocaleString('en-US') + ' · ' + event.title;
      return option;
    }));
    const render = () => {
      const index = Number(select.value);
      const event = events[index];
      detail.textContent = event.detail;
      source.href = event.source;
      previous.disabled = index === 0;
      next.disabled = index === events.length - 1;
    };
    select.addEventListener('change', render);
    previous.addEventListener('click', () => { select.selectedIndex -= 1; render(); });
    next.addEventListener('click', () => { select.selectedIndex += 1; render(); });
    render();
  }
  const progress = document.getElementById('train-lr-progress');
  const progressLabel = document.getElementById('train-lr-progress-label');
  const lrOutput = document.getElementById('train-lr-output');
  if (progress && progressLabel && lrOutput) {
    const render = () => {
      const p = Number(progress.value);
      const ratio = p <= 1 ? p : 1 - 0.95 * (p - 1) / 99;
      progressLabel.textContent = p.toFixed(1).replace(/\.0$/, '') + '%';
      lrOutput.textContent = '归一化学习率约 ' + (ratio * 100).toFixed(1) + '% peak；MuonH／AdamH 约 ' + (0.003291 * ratio).toFixed(6) + '，Adam 约 ' + (0.000759 * ratio).toFixed(6) + '。';
    };
    progress.addEventListener('input', render);
    render();
  }
})();
