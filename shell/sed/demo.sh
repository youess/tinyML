#!/bin/bash

# 这个脚本是照搬https://coolshell.cn/articles/9104.html的内容，
# 手敲一遍更加容易理解。
# 尤其通过前面简单的例子，大致对sed有一个了解，
# 后面关于sed流处理的伪代码，醍醐灌顶，对sed有了更加深入的理解。

cat << EOF > pets.txt
This is my cat
  my cat's name is betty
This is my dog
  my dog's name is frank
This is my fish
  my fish's name is george
This is my goat
  my goat's name is adam
EOF

## s命令进行替换
# 全部替换my为denis's, 匹配最好用双引号
sed "s/my/denis's/g" pets.txt

# 每行添加内容, 行首和行尾
sed "s/^/#/g" pets.txt
sed "s/$/ --- /g" pets.txt

cat << EOF > html.txt
<b>This</b> is what <span style="text-decoration: underline;">I</span> meant. Understand?
EOF

# wrong eg
sed "s/<.*>//g" html.txt

# correct one, []中^表示除了这个范围里面的符号
sed "s/<[^>]*>//g" html.txt

# 指定第三行进行替换
sed "3s/my/your/g" pets.txt

# 替换一个范围的文本
sed "3,6s/my/your/g" pets.txt

cat <<EOF > my.txt
This is my cat, my cat's name is betty
This is my dog, my dog's name is frank
This is my fish, my fish's name is george
This is my goat, my goat's name is adam
EOF

# 替换每一行的第一个
sed "s/s/S/1" my.txt

# 替换每一行的第三个及以后
sed "s/s/S/3g" my.txt

## 多个匹配; 多个模式用分号隔开
sed "1,3s/my/your/g; 3,\$s/This/That/g" my.txt
# 等价于
sed -e "1,3s/my/your/g" -e "3,\$s/This/That/g" my.txt

# &当作匹配量
sed "s/my/[&]/g" my.txt

## 圆括号匹配, 圆括号匹配的字符串可以当成变量来使用，变量是\1, \2
sed "s/This is my \([^,&]*\),.*is \(.*\)/\1:\2/g" my.txt

## N命令, 相当于一次读入两行，然后sed看作是一行
sed "N;s/my/your/" pets.txt

# 解释N命令
sed "N;s/\n/,/" pets.txt

## a命令和i命令, 用于添加行, 一个是在之后，一个是在前
# 在第一行之前添加一行
sed "1 i This is my monkey, my monkey's name is wukong" my.txt

# 在最后一行添加
sed "$ a This is my monkey, my monkey's name is wukong" my.txt

# 匹配添加
sed "/fish/a This is my monkey, my monkey's name is wukong" my.txt

# 没有就是对每一行
sed "/my/a -----" my.txt

## c命令 进行替换
# 替换第二行为
sed "2 c This is my monkey, my monkey's name is wukong" my.txt

# 替换匹配行
sed "/fish/ c This is my monkey, my monkey's name is wukong" my.txt

## 删除命令d
# 删除匹配fish的行
sed "/fish/d" my.txt

# 删除第二行
sed "2d" my.txt

# 范围删除
sed '2,$d' my.txt         # 单引号不用加转义，但是需要转义的时候就必须用双引号

## 打印命令p 
# 打印全部行，并重复打印了匹配行
sed "/fish/p" my.txt

# 只打印匹配行, 添加参数-n
sed -n "/fish/p" my.txt

# 多个匹配模式打印, 多个模式用,隔开，不是之前的分号;
sed -n "/dog/,/fish/p" my.txt

# 从第2行开始打印到匹配行
sed -n "2,/fish/p" my.txt
 
########################
# 几个sed的知识点
########################

## 模式空间pattern space

#  SED 批处理逻辑
#  foreach line in file {
#      //读一行就放入Pattern_Space
#      Pattern_Space <= line;
#   
#      // 对每个pattern space执行sed命令
#      Pattern_Space <= EXEC(sed_cmd, Pattern_Space);
#   
#      // 如果没有指定 -n 则输出处理后的Pattern_Space
#      if (sed option hasn't "-n")  {
#         print Pattern_Space
#      }
#  }

## 地址Address
# 引号中的内容一般都是:  [address[,address]][!]{cmd}
# !表示在找到的地址行不执行命令

# 命令执行的伪代码
# bool bexec = false
# foreach line in file {
#     if ( match(address1) ){
#         bexec = true;
#     }
#  
#     if ( bexec == true) {
#         EXEC(sed_cmd);
#     }
#  
#     if ( match (address2) ) {
#         bexec = false;
#     }
# }

# 使用相对位置的address
sed "/dog/,+3 s/^/# /g" pets.txt

## 命令打包, 用{}
# 3到6行address 执行{}中匹配的命令
sed "3,6 {/This/d}" pets.txt

# 匹配成功之后再匹配
sed "{/This/{/fish/d}}" pets.txt

# 对每一行match1 then do1 and match2 then do2
sed "{/This/d; s/^ *//g}" pets.txt

## HoldSpace, 初始空间里面是一个换行符
## 与之相关的命令包括
# g, 将hold space内容拷贝到pattern space中，原来pattern space内容清除
# G, 将hold space内容append到pattern space\n后
# h, 将pattern space内容拷贝到hold space中， 原来hold space内容清除
# H, 将pattern space内容append到hold space\n后
# x, 交换pattern space和hold space内容

cat << EOF > t.txt
one
two
three
EOF

# 示例1, 先H再g，两个命令
sed "H;g" t.txt   # 参考图进行理解

# 反序一个文件的行
sed "1!G;h;\$!d" t.txt

# 总结来说pattern space里的内容才打印，hold space里面的不会被打印


