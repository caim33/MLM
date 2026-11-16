(() => {
  const $ = (selector) => document.querySelector(selector);
  const chapters = [...document.querySelectorAll('.chapter')];
  const links = [...document.querySelectorAll('.sidebar nav a')];
  const search = $('#chapter-search');
  const searchText = new Map(chapters.map(section => ['#' + section.id, section.textContent.toLowerCase()]));
  search.addEventListener('input', () => {
    const term = search.value.trim().toLowerCase();
    let matches = 0;
    for (const link of links) {
      const match = !term || (searchText.get(link.hash) || link.textContent).includes(term);
      link.hidden = !match;
      if (match) matches++;
    }
    $('#search-status').textContent = term ? `找到 ${matches} 个相关章节；点击目录进入。` : '';
  });
  links.forEach(link => link.addEventListener('click', () => {
    if (window.matchMedia('(max-width:780px)').matches) $('.sidebar details').open = false;
  }));
  if (window.matchMedia('(max-width:780px)').matches) $('.sidebar details').open = false;
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top-b.boundingClientRect.top);
    if (!visible.length) return;
    links.forEach(link => {
      const active = link.hash === '#' + visible[0].target.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }, {rootMargin:'-80px 0px -60% 0px',threshold:0});
  chapters.forEach(section => observer.observe(section));
  const updateProgress = () => {
    const range = document.documentElement.scrollHeight-window.innerHeight;
    $('#reading-progress').style.width = (range > 0 ? Math.min(100, Math.max(0,window.scrollY/range*100)) : 0) + '%';
  };
  window.addEventListener('scroll', updateProgress, {passive:true});
  window.addEventListener('resize', updateProgress);
  updateProgress();
  $('#print-report').addEventListener('click', () => window.print());
  const dialog = $('#image-dialog');
  document.querySelectorAll('.zoom-image').forEach(button => button.addEventListener('click', () => {
    $('#dialog-image').src = button.dataset.src;
    $('#dialog-image').alt = button.dataset.caption;
    $('#dialog-caption').textContent = button.dataset.caption;
    dialog.showModal();
  }));
  dialog.addEventListener('click', event => {if (event.target === dialog) dialog.close();});

  const updateKV = () => {
    const length = Number($('#kv-length').value);
    const batch = Math.min(64, Math.max(1, Number($('#kv-batch').value) || 1));
    const tokens = length*1024;
    const base = batch*tokens*61*2/(2**30);
    $('#kv-length-label').textContent = length + ' Ki tokens';
    $('#kv-mha').textContent = (base*2*128*128).toFixed(2)+' GiB';
    $('#kv-mla').textContent = (base*(512+64)).toFixed(2)+' GiB';
  };
  $('#kv-length').addEventListener('input', updateKV);
  $('#kv-batch').addEventListener('input', updateKV);
  updateKV();
  const rewards = [...document.querySelectorAll('#reward-buttons button')];
  const updateRewards = () => {
    const values = rewards.map(button => Number(button.dataset.reward));
    const mean = values.reduce((a,b) => a+b,0)/values.length;
    const std = Math.sqrt(values.reduce((a,b) => a+(b-mean)**2,0)/values.length);
    $('#reward-mean').textContent = mean.toFixed(2);
    $('#reward-std').textContent = std.toFixed(2);
    $('#reward-advantages').textContent = '优势：['+values.map(v => {
      const a=(v-mean)/(std+1e-8);return (a>0?'+':'')+a.toFixed(2);
    }).join(', ')+']';
    $('#reward-reading').textContent = std === 0 ? '组内奖励全相同：结果奖励的相对优势全为0。这不等于完整损失没有梯度，KL等项仍可能参与更新。' : '有成有败时，结果奖励提供相对区分。优势表示相对本组的好坏，不是轨迹的绝对质量。';
  };
  rewards.forEach((button,index) => {
    button.setAttribute('aria-pressed',button.dataset.reward==='1'?'true':'false');
    button.addEventListener('click', () => {
      button.dataset.reward = button.dataset.reward==='1'?'0':'1';
      button.textContent = `轨迹${index+1}：${button.dataset.reward}`;
      button.setAttribute('aria-pressed',button.dataset.reward==='1'?'true':'false');
      updateRewards();
    });
  });
  updateRewards();
  let step = 0;
  const parallelSteps = [
    ['① FSDP：同一列收集当前层所需的专家参数', 'GPU0与GPU2各保存E0/E1的一部分参数，在计算前收集成当前层完整的E0/E1；GPU1与GPU3对E2/E3同样处理。参数沿列移动，目的是让当前设备能执行它负责的专家。参数不必永久完整驻留。'],
    ['② EP：每一行把token激活送到选中的专家', '假设GPU0的token选中E3，就在EP组A里把这份激活送给GPU1；GPU2的数据只在组B里对应路由到GPU3。这里移动的是token激活与路由信息，专家身份和分组保持不变。'],
    ['③ 专家计算，再沿EP组返回并合并', 'GPU1用刚收集的E3权重算出输出，再把结果返回GPU0。若token选中多个专家，来源设备按对应门控权重合并。两个EP组可处理各自的输入批次；重叠能否提速需实际测量。'],
    ['④ 反向传播，FSDP归约并分片保存梯度', '反向沿EP组传回相关梯度；同一专家FSDP组聚合各批次的贡献并按分片保存梯度、更新状态。FSDP解决参数及训练状态的持有方式，EP解决专家计算的归属与token分发，所以可以组合。']
  ];
  const updateParallel = () => {
    $('.parallel-lab').dataset.step = String(step);
    $('#parallel-count').textContent = `${step+1} / ${parallelSteps.length}`;
    $('#parallel-heading').textContent = parallelSteps[step][0];
    $('#parallel-text').textContent = parallelSteps[step][1];
    $('#parallel-prev').disabled = step===0;
    $('#parallel-next').disabled = step===parallelSteps.length-1;
  };
  $('#parallel-prev').addEventListener('click',() => {step=Math.max(0,step-1);updateParallel();});
  $('#parallel-next').addEventListener('click',() => {step=Math.min(parallelSteps.length-1,step+1);updateParallel();});
  updateParallel();
})();
