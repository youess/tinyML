#!/bin/bash

# 同样的awk note照抄https://coolshell.cn/articles/9070.html, 请看原文

# 数据
# cat netstat.txt

## 入门, 只能用单引号'和{}进行命令表达
awk '{print $1, $4}' netstat.txt

# 格式化输出
awk '{printf "%-8s %-8s %-8s %-18s %-22s %-15s\n",$1,$2,$3,$4,$5,$6}' netstat.txt

# 过滤记录
awk ' $3==0 && $6 == "LISTEN" ' netstat.txt

awk '$3>0 {print $0}' netstat.txt

# 包括第一行
awk '$3==0 && $6=="LISTEN" || NR==1' netstat.txt

# 加格式化
awk '$3==0 && $6=="LISTEN" || NR==1 {printf "%-20s %-20s %s\n",$4,$5,$6}' netstat.txt

## 内建变量
# $0	         当前记录（这个变量中存放着整个行的内容）
# $1~$n	         当前记录的第n个字段，字段间由FS分隔
# FS	         输入字段分隔符 默认是空格或Tab
# NF	         当前记录中的字段个数，就是有多少列
# NR	         已经读出的记录数，就是行号，从1开始，如果有多个文件话，这个值也是不断累加中。
# FNR	         当前记录数，与NR不同的是，这个值会是各个文件自己的行号
# RS	         输入的记录分隔符， 默认为换行符
# OFS	         输出字段分隔符， 默认也是空格
# ORS	         输出的记录分隔符，默认为换行符
# FILENAME	 当前输入文件的名字

awk '$3==0 && $6=="ESTABLISHED" || NR==1 {printf "%02s %02s %-20s %-25s %s\n",NR, FNR, $4,$5,$6}' netstat.txt

# 指定分隔符, "-"表示左对齐
# awk 'BEGIN{FS=":"} {printf "%10s %-5s %s\n", $1,$3,$6}' /etc/passwd | head

# 等价
awk -F: '{print $1,$3,$6}' /etc/passwd | head

# 多个分割符用
# awk -F '[;:]'

# 指定输出分隔符, 注意OFS变量，类似环境变量
awk -F: '{print $1,$3,$6}' OFS='\t' /etc/passwd | head

## 字符串匹配, ~表示模式开始 // -> 模式
awk '$6 ~ /FIN/ || NR == 1 {print NR,$4,$5,$6}' OFS='\t' netstat.txt

# 模式 or
awk '$6 ~ /FIN|TIME/ || NR == 1 {print NR, $4, $5, $6}' OFS="\t" netstat.txt

# 模式取反
awk '$6 !~ /FIN|TIME/ || NR == 1 {print NR, $4, $5, $6}' OFS="\t" netstat.txt

## 拆分文件

# 按照第六列分割文件, ls -> ESTABLISHED  FIN_WAIT1  FIN_WAIT2  LAST_ACK  LISTEN  netstat.txt  TIME_WAIT
awk 'NR!=1{print > $6}' netstat.txt

# 指定列输出到拆分文件
awk 'NR!=1{print $4,$5 > $6}' netstat.txt

# 更加复杂一点
awk 'NR != 1 { if ($6 ~ /TIME|ESTABLISHED/) print > "1.txt"; else if ($6 ~ /LISTEN/) print > "2.txt"; else print > "3.txt" }' netstat.txt

## 统计
ls -l *.txt | awk '{sum+=$5}END{print sum / 1024, "KB"}'

# 分组统计connection状态个数
awk 'NR!=1 {a[$6]++} END { for (i in a) print i, ",", a[i]; }' netstat.txt

# 查看每个用户进程占用内存
ps aux | awk 'NR!=1{a[$1]+=$6;} END { for(i in a) print i ", " a[i]"KB";}'

cat << EOF > score.txt
Marry   2143 78 84 77
Jack    2321 66 78 45
Tom     2122 48 77 71
Mike    2537 87 97 95
Bob     2415 40 57 62
EOF

## 环境变量交互
x=5
export y=10
awk -v val=$x '{print $1,$2,$3, $4+val, $5+ENVIRON["y"]}' OFS="\t" score.txt

# 查看文件中长度大于80的行
awk 'length>80' netstat.txt

