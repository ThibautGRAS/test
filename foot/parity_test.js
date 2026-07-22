const fs=require('fs');
const M=JSON.parse(fs.readFileSync('model.json'));
const FIX=JSON.parse(fs.readFileSync('fixtures.json'));
const NEWS=JSON.parse(fs.readFileSync('news.json'));
const src=fs.readFileSync('/tmp/prod.js','utf8');
const head=src.split('/* ---------- vue journées')[0].replace(/^'use strict';/,'').replace(/let M=null.*$/m,'');
const fn=new Function('M','FIX','NEWS', head+'; buildRanks(); return computeMatch;');
const cm=fn(M,FIX,NEWS);
const tests=[['Paris SG','Rennes'],['Le Mans','Brest'],['Marseille','Strasbourg'],
             ['Angers','Lille'],['Lens','Auxerre'],['Nice','Lorient'],
             ['Toulouse','Lyon'],['Troyes','Paris FC']];
const out={};
tests.forEach(([h,a])=>{
  const p=cm(h,a,{news:false,rh:7,ra:7});
  out[h+'|'+a]={pb:p.pb.map(x=>+x.toFixed(4)), score:p.bi+'-'+p.bj, xg:[+p.xgH.toFixed(3),+p.xgA.toFixed(3)]};
});
console.log(JSON.stringify(out));
