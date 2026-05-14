# 数据收集
现在需要为 @data/pair_merged_data/all_pair_data.tsv 中每一个反应寻找到对应的 别名反应 文本，打算使用如下方式
## 检索逻辑（框架—）
1. 先找到这条 pair 对应哪个源记录
  当前有 3 条匹配路径：

  - brenda_by_enz
      - 先拿 all_pair_data.tsv 里的 (enz_id, std_rxn)。
      - 在 Brenda 源里按同一个 enz_id 找，命中同一个 std_rxn 就直接用。
      - 这是最优先路径，因为 Brenda 自带 RXN_TEXT。
  - rhea_by_enz
      - 如果 Brenda 没命中，就去 Rhea 源里按同一个 enz_id + std_rxn 找。
      - 这个路径通常只有 Rhea_ID，没有 RXN_TEXT，后面还要继续补文本。
  - *_by_std
      - 如果按 enz_id 都没找到，就退化成只按 std_rxn 在全局源库里找。
      - 可能命中 Brenda，也可能命中 Rhea。
      - 这是“同反应全局兜底”，不再要求同一个酶。
2. 
找到源记录后，怎么得到 RXN_TEXT
  当前实际存在 5 条文本路径：

  - 路径 A：直接使用 Brenda 自带 RXN_TEXT
      - 命中 brenda_by_enz 或 brenda_by_std 时，如果原记录已有 RXN_TEXT，直接写出。
  - 路径 B：用同一个 std_rxn 的 Brenda 文本补给 Rhea
      - 如果命中的是 Rhea 记录，但它没有 RXN_TEXT，脚本会先看这个 std_rxn 是否在 Brenda 里出现过带文本的版本。
  - 路径 C：用 rhea_name_smiles.txt 逐分子翻译
      - 如果还没有文本，就把 std_rxn 拆成单分子。
      - 每个 canonical SMILES 去 task4/data/rhea_name_smiles.txt 里查别名。
      - 全命中时，拼成 A + B = C + D。
      - 没有全命中，则进入下一路径
  - 路径 E：按 Rhea_ID 远程补文本
      - 只有当上一步出现 used_fallback=True 且记录里有 Rhea_ID 时，才会触发。
3. 
Rhea_ID 这条远程路径内部，其实又有几层
  当前代码写的是：
  - E1：Rhea_ID -> /json -> equation
      - 仍然优先，直接返回 equation
  - E2：Rhea_ID -> /json -> left/right -> label
      - 不再读不存在的 participants
      - 改为分别读取 left 和 right
      - 每侧把 label 提出来，拼成 A + B = C + D
      
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


步骤：
1. 将 @task4/data/brenda_rxntxt_uniprot_washed.tsv 和 @task4/data/cleaned_rhea_uniprot_washed.tsv 以及 @task4/data/pair_merged_data/all_pair_data.tsv 中 mapped_rxn 使用 RDKit 工具删除 atom mapping 序号，并标准化（要求反应物顺序，以及产物顺序也唯一），结果对应新的一列 std_rxn
2. 按照框架进行检索得到 RXN_TEXT
3. 需要打印处理过程中各环节处理成功的反应数目

