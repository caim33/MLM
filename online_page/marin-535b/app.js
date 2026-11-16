(() => {
 const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
 const progress=$('#reading-progress');
 const update=()=>{const max=document.documentElement.scrollHeight-innerHeight;progress.style.width=(max>0?100*scrollY/max:0)+'%';};
 addEventListener('scroll',update,{passive:true});addEventListener('resize',update);update();
 const menu=$('#menu-toggle');menu?.addEventListener('click',()=>{const open=$('#toc').classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
 $$('#toc a[href^="#"]').forEach(a=>a.addEventListener('click',()=>{$('#toc').classList.remove('open');menu?.setAttribute('aria-expanded','false');}));
 const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(!visible.length)return;$$('#toc nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+visible[0].target.id));},{rootMargin:'-10% 0px -65% 0px',threshold:0});
 $$('.report-section').forEach(s=>observer.observe(s));
 $('#back-top').addEventListener('click',()=>scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'}));
 $('#print-report').addEventListener('click',()=>window.print());
 const dialog=$('#image-dialog');
 document.addEventListener('click',e=>{const btn=e.target.closest('[data-zoom]');if(!btn)return;e.preventDefault();const fig=btn.closest('figure'),img=fig?.querySelector('img');dialog.querySelector('img').src=img?.src||btn.dataset.zoom;dialog.querySelector('img').alt=img?.alt||'配图放大';dialog.querySelector('p').textContent=img?.alt||'';dialog.showModal();});
 dialog.querySelector('button').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
 const index=$('#figure-index');
 $$('figure').forEach((fig,i)=>{fig.id||=`figure-${i+1}`;let caption=fig.querySelector('figcaption');if(!caption){caption=document.createElement('figcaption');fig.append(caption);}let num=document.createElement('span');num.className='figure-num';num.textContent=`FIG. ${String(i+1).padStart(2,'0')}`;caption.prepend(num);const img=fig.querySelector('img');if(img&&!fig.querySelector('[data-zoom]')){const b=document.createElement('button');b.className='btn zoom-btn';b.dataset.zoom=img.getAttribute('src');b.textContent='放大查看 ⤢';caption.append(b);}if(index){const li=document.createElement('li'),a=document.createElement('a');a.href='#'+fig.id;a.textContent=`${String(i+1).padStart(2,'0')} · ${img?.alt||fig.dataset.title||fig.querySelector('h3')?.textContent||caption.querySelector('strong')?.textContent||fig.querySelector('svg')?.getAttribute('aria-label')||'交互图解'}`;li.append(a);index.append(li);}});
 const count=$('#total-figures');if(count)count.textContent=$$('figure').length;
 document.querySelectorAll('a[target="_blank"]').forEach(a=>a.rel='noopener noreferrer');
})();
