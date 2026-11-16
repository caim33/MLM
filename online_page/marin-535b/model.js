(function () {
  'use strict';
  function bootModel() {
    var demo = document.getElementById('model-router-demo');
    if (demo) {
      var token = document.getElementById('model-token');
      var drop = document.getElementById('model-route-drop');
      var next = document.getElementById('model-route-next');
      var reset = document.getElementById('model-route-reset');
      var experts = document.getElementById('model-route-experts');
      var note = document.getElementById('model-route-note');
      var samples = [[1.8, 0.9, 0.3, -0.2], [-0.6, 0.1, 1.1, 1.9], [0.3, 1.7, 1.1, -0.5]];
      var biases = [0, 0.2, -0.1, 0.1];
      var stage = 0;
      function renderRoute() {
        var raw = samples[Number(token.value)];
        var ranked = raw.map(function (score, id) { return {id:id, score:score + biases[id]}; }).sort(function(a,b){return b.score-a.score;});
        var selected = [ranked[0].id, ranked[1].id];
        var sig = selected.map(function(id){return 1/(1+Math.exp(-raw[id]));});
        var weights = sig.map(function(s){return s/(sig[0]+sig[1])*2.5;});
        experts.innerHTML = raw.map(function(score,id){
          var index = selected.indexOf(id);
          var active = stage >= 1 && index >= 0;
          var lost = active && stage >= 2 && drop.checked && index === 1;
          var status = stage === 0 ? '等待 router' : '未选中';
          if (active) status = '选中 · 权重 ' + weights[index].toFixed(2);
          if (active && stage === 2) status = lost ? '分配溢出 · 本次不计算' : '执行专家变换';
          if (active && stage === 3) status = lost ? '该份贡献缺失' : '返回加权输出 · ' + weights[index].toFixed(2);
          return '<div class="model-route-expert' + (active ? ' selected':'') + (lost ? ' dropped':'') + '"><strong>E'+id+'</strong><small>原始分数 '+score.toFixed(1)+' / 偏置 '+(biases[id]>=0?'+':'')+biases[id].toFixed(1)+'</small><small>'+status+'</small></div>';
        }).join('');
        var messages = [
          '① 输入：一个 token 的向量进入该层。真实模型向量宽度为 6,144；共享专家和残差路径始终保留。',
          '② 选择：按分数加教学偏置，选择 E'+selected[0]+'、E'+selected[1]+'。用原始分数的 sigmoid 计算合并权重，再缩放为总和 2.5。此例没有模拟真实 QB 阈值估计。',
          drop.checked ? '③ 派发与计算：发给 E'+selected[1]+' 的一份 assignment 被容量限制丢弃；E'+selected[0]+' 仍计算，输入 token 仍在。' : '③ 派发与计算：同一 token 的表示交给两个被选专家，各自产生变换后的表示。真实 EP 会跨 GPU 搬运这些激活。',
          drop.checked ? '④ 合并：只合并存活的路由贡献，再加共享专家与残差。这就是 assignment drop 不等于整个 token 被删除；此教学例不重新归一存活权重。' : '④ 合并：将两个路由专家输出按权重相加，再与共享专家输出和残差共同更新 token。真实模型每 token 选 8 / 384，而此图为 2 / 4。'
        ];
        note.textContent = messages[stage];
        next.textContent = stage === 3 ? '再走一遍' : '下一步 → ' + (stage + 2);
      }
      token.addEventListener('change',function(){stage=0;renderRoute();});
      drop.addEventListener('change',renderRoute);
      next.addEventListener('click',function(){stage=(stage+1)%4;renderRoute();});
      reset.addEventListener('click',function(){stage=0;renderRoute();});
      renderRoute();
    }
    var calc = document.getElementById('model-comm-calculator');
    if (calc) {
      var batch = document.getElementById('model-comm-batch');
      var count = document.getElementById('model-comm-experts');
      var topk = document.getElementById('model-comm-topk');
      var latent = document.getElementById('model-comm-latent');
      function calculate() {
        var B=Number(batch.value),E=Number(count.value),K=Number(topk.value),L=Number(latent.value),P=64,I=3072;
        var T=B*4096/P;
        var fsdp=3*E*L*I*2*(P-1)/P/Math.pow(1024,3);
        var ep=2*T*K*L*2*(P-1)/P/Math.pow(1024,3);
        document.getElementById('model-comm-tokens').textContent=T.toLocaleString('en-US');
        document.getElementById('model-comm-fsdp').textContent=fsdp.toFixed(2)+' GiB';
        document.getElementById('model-comm-ep').textContent=ep.toFixed(2)+' GiB';
        var maximum=Math.max(fsdp,ep);
        document.getElementById('model-fsdp-bar').style.width=(fsdp/maximum*82)+'%';
        document.getElementById('model-ep-bar').style.width=(ep/maximum*82)+'%';
        document.getElementById('model-comm-note').textContent='当前每 GPU '+(T*K).toLocaleString('en-US')+' 份 assignment。FSDP / EP 负载约 '+(fsdp/ep).toFixed(2)+' 倍。E 是专家总数，K 是 top-k，L 是路由宽度；改变这些变量是对比假设，不代表已训练的其他 Marin 配置。';
      }
      [batch,count,topk,latent].forEach(function(input){input.addEventListener('change',calculate);});
      calculate();
    }
  }
  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',bootModel);
  else bootModel();
})();
