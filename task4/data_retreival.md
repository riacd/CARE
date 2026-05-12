# 数据收集
现在需要为 @task4/data/pair_merged_data/all_pair_data.tsv 中每一个反应寻找到对应的 别名反应 文本，打算使用如下方式
## 找到每个反应在原始数据库中的索引
由于文件 @task4/data/pair_merged_data/all_pair_data.tsv 中的所有反应，全部来自于两个数据源
1. @task4/data/brenda_rxntxt_uniprot_washed.tsv
2. @task4/data/cleaned_rhea_uniprot_washed.tsv
通过寻找每条反应在数据源中对应的条目，可以检索获得别名反应的名称

结果文件保存如下列：
1. RXN_TEXT：反应的别名表达式
2. Rhea_ID：有的话则有，没有则为NaN
3. std_rxn：标准反应 SMILES 表达式
4. enz_id：Uniprot ID
未匹配数据文件保存如下列：
1. enz_id
2. std_rxn


检索方式：
0. 将 @task4/data/brenda_rxntxt_uniprot_washed.tsv 和 @task4/data/cleaned_rhea_uniprot_washed.tsv 以及 @task4/data/pair_merged_data/all_pair_data.tsv 中 mapped_rxn 使用 RDKit 工具删除 atom mapping 序号，并标准化（要求反应物顺序，以及产物顺序也唯一），结果对应新的一列 std_rxn
1. 根据 Uniprot ID 来检索，将 @task4/data/brenda_rxntxt_uniprot_washed.tsv 和 @task4/data/cleaned_rhea_uniprot_washed.tsv 分别各自转化成 enz_id -> rxn(Rhea_ID/RXN_TEXT, std_rxn) 的一对多映射，保存为字典
2. 依次使用 @task4/data/pair_merged_data/all_pair_data.tsv 中每条数据 (enz_id, std_rxn) 对应的 enz_id 进行检索
3. 先使用根据 @task4/data/brenda_rxntxt_uniprot_washed.tsv 构造的字典，使用 enz_id 进行检索得到若干反应，第一个与 std_rxn 匹配的反应数据  (RXN_TEXT, Rhea_ID, std_rxn, enz_id) 放入结果文件中，没有匹配到的原始数据保存到未匹配数据文件中
4. 使用根据 @task4/data/cleaned_rhea_uniprot_washed.tsv 构造的字典，使用 enz_id 进行检索得到若干反应，第一个与 std_rxn 匹配的反应数据  (RXN_TEXT, Rhea_ID, std_rxn, enz_id) 中，有Rhea_ID，但缺乏RXN_TEXT 数据，下一步就是根据std_rxn构造相应的RXN_TEXT
5. 对于步骤4中获得的 缺乏RXN_TEXT 数据 的条目以及 未匹配数据文件 中所有条目，使用 @task4/data/rhea_name_smiles.txt 中描述的SMILES到别名到转化方式，将原本的SMILES 序列转化为 别名的反应描述作为 RXN_TEXT 

## 实现
已实现脚本：@task4/retrieve_reaction_aliases.py

执行命令：
```bash
python task4/retrieve_reaction_aliases.py
```

脚本按如下规则执行：
1. 对 Brenda、Rhea、all_pair_data 中的 `mapped_rxn` 使用 RDKit 去除 atom mapping，并将反应物集合、产物集合分别做 canonical SMILES 和字典序排序，得到唯一的 `std_rxn`
2. 将 Brenda 数据构造成 `enz_id -> [(RXN_TEXT, '', std_rxn, enz_id)]`
3. 将 Rhea 数据构造成 `enz_id -> [('', Rhea_ID, std_rxn, enz_id)]`
4. 对 `all_pair_data.tsv` 中每条 `(enz_id, std_rxn)`，优先按 Brenda 的 `(enz_id, std_rxn)` 精确匹配
5. 若 Brenda 未命中，再按 Rhea 的 `(enz_id, std_rxn)` 精确匹配
6. 若仍未命中，则按 `std_rxn` 在全局源库中兜底匹配
7. 若命中的是 Rhea 且 `RXN_TEXT` 为空，则先尝试用同一个 `std_rxn` 在 Brenda 中补 `RXN_TEXT`
8. 若仍然没有 `RXN_TEXT`，则使用 `rhea_name_smiles.txt` 将 `std_rxn` 拆成单分子并替换成别名，最终拼接为 `reactant1 + reactant2 = product1 + product2` 形式的反应文本
9. 若某些分子在 `rhea_name_smiles.txt` 中没有别名，则仅对这些分子保留 canonical SMILES，其余分子仍使用别名

## 输出文件
结果文件：
1. @task4/data/pair_merged_data/reaction_aliases.tsv

未匹配文件：
1. @task4/data/pair_merged_data/reaction_aliases_unmatched.tsv

运行元数据：
1. @task4/data/pair_merged_data/reaction_aliases.metadata.json

## 结果统计
当前一次完整运行的统计如下：
1. `all_pair_data.tsv` 共 265,920 条 pair，全部完成匹配
2. 其中 `brenda_by_enz` 直接命中 34,084 条
3. 其中 `rhea_by_enz_plus_brenda_text` 通过同反应 Brenda 文本补到 `RXN_TEXT` 的有 1,080 条
4. 其中 `rhea_by_enz_plus_rhea_name_smiles` 通过 `rhea_name_smiles.txt` 生成 `RXN_TEXT` 的有 225,895 条
5. 其中 `rhea_by_std_plus_rhea_name_smiles` 通过 `std_rxn` 全局兜底后再生成 `RXN_TEXT` 的有 4,861 条
6. 使用 `rhea_name_smiles.txt` 生成文本时，169,805 条反应实现了全分子别名覆盖
7. 使用 `rhea_name_smiles.txt` 生成文本时，60,951 条反应存在部分分子未命中映射，因此局部保留了 canonical SMILES
8. 最终 265,920 条结果全部带有 `RXN_TEXT`
9. `reaction_aliases_unmatched.tsv` 当前仅包含表头，说明没有未匹配 pair

## 备注
1. Rhea 源数据本身不包含 `RXN_TEXT` 列，因此需要第 5 步额外使用 `rhea_name_smiles.txt` 按分子构造反应文本
2. Brenda 源文件中有 33 行缺少 `Uniprot ID`，这些行不会进入 `(enz_id, std_rxn)` 索引，已记录在 `reaction_aliases.metadata.json`
