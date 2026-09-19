/***********************************
	本驱动文件仅适配HAL库版本
***********************************/
#include "stm32f4xx_hal.h"
#include "OLED_Font.h"
#include "oled.h"
#include "i2c.h"

/* OLED 使用 I2C1 */
#define i2c_set hi2c1

/* SSD1306 初始化命令序列（与参考代码完全一致，27字节）
 * BUG修复：原 0xD8,0x30 改为 0xDB,0x30（VCOMH去选电平）
 * 0xD8 不是 SSD1306 有效命令，0xDB 才是 */
uint8_t CMD_Data[]={
0xAE,       /*  0: display off */
0x00,       /*  1: set low column address */
0x10,       /*  2: set high column address */
0x40,       /*  3: set start line address */
0xB0,       /*  4: set page address */
0x81, 0xFF, /*  5-6: contrast control = 255 */
0xA1,       /*  7: set segment remap */
0xA6,       /*  8: normal display */
0xA8, 0x3F, /*  9-10: set multiplex ratio = 63 */
0xC8,       /* 11: COM scan direction */
0xD3, 0x00, /* 12-13: set display offset = 0 */
0xD5, 0x80, /* 14-15: set osc division */
0xD8, 0x05, /* 16-17: set area color mode off */
0xD9, 0xF1, /* 18-19: Set Pre-Charge Period */
0xDA, 0x12, /* 20-21: set COM pin configuration */
0xDB, 0x30, /* 22-23: set Vcomh (修复: 原为 0xD8) */
0x8D, 0x14, /* 24-25: charge pump enable */
0xAF        /* 26: turn on oled panel */
};


void WriteCmd(void)
{
	uint8_t i = 0;
	for(i=0; i<sizeof(CMD_Data); i++)
	{
		HAL_I2C_Mem_Write(&i2c_set ,0x78,0x00,I2C_MEMADD_SIZE_8BIT,CMD_Data+i,1,0x100);
	}
}

void OLED_WR_CMD(uint8_t cmd)
{
	HAL_I2C_Mem_Write(&i2c_set ,0x78,0x00,I2C_MEMADD_SIZE_8BIT,&cmd,1,0x100);
}

void OLED_WR_DATA(uint8_t data)
{
	HAL_I2C_Mem_Write(&i2c_set ,0x78,0x40,I2C_MEMADD_SIZE_8BIT,&data,1,0x100);
}

void OLED_Init(void)
{
	HAL_Delay(200);
	OLED_Clear();
	WriteCmd();
}

void OLED_Clear(void)
{
	uint8_t i,n;
	for(i=0;i<8;i++)
	{
		OLED_WR_CMD(0xb0+i);
		OLED_WR_CMD (0x00);
		OLED_WR_CMD (0x10);
		for(n=0;n<128;n++)
			OLED_WR_DATA(0);
	}
}

void OLED_Display_On(void)
{
	OLED_WR_CMD(0X8D);
	OLED_WR_CMD(0X14);
	OLED_WR_CMD(0XAF);
}

void OLED_Display_Off(void)
{
	OLED_WR_CMD(0X8D);
	OLED_WR_CMD(0X10);
	OLED_WR_CMD(0XAE);
}

void OLED_Set_Pos(uint8_t x, uint8_t y)
{
	OLED_WR_CMD(0xb0+y);
	OLED_WR_CMD(((x&0xf0)>>4)|0x10);
	OLED_WR_CMD(x&0x0f);
}

void OLED_On(void)
{
	uint8_t i,n;
	for(i=0;i<8;i++)
	{
		OLED_WR_CMD(0xb0+i);
		OLED_WR_CMD(0x00);
		OLED_WR_CMD(0x10);
		for(n=0;n<128;n++)
			OLED_WR_DATA(1);
	}
}

unsigned int oled_pow(uint8_t m,uint8_t n)
{
	unsigned int result=1;
	while(n--)result*=m;
	return result;
}

void OLED_ShowRectangle(uint8_t x,uint8_t y,uint8_t high)
{
	int n;
	OLED_Set_Pos(x,y);
	for(n=0;n<high;n++)
	{
  OLED_WR_DATA(0xff);
	}
}

void OLED_ShowSignedNum(uint8_t x,uint8_t y,int num,uint8_t len,uint8_t size2)
{
	uint8_t t,temp;
	uint8_t enshow=0;	if(num>0)
	{
		OLED_ShowChar(x,y,'+',12);
	}
	else if(num<0)
	{
		OLED_ShowChar(x,y,'-',12);
		num=-num;
	}
	for(t=0;t<len;t++)
	{
		temp=(num/oled_pow(10,len-t-1))%10;
		if(enshow==0&&t<(len-1))
		{
			if(temp==0)
			{
				OLED_ShowChar(6+x+(size2/2)*t,y,' ',size2);
				continue;
			}else enshow=1;

		}
	 	OLED_ShowChar(6+x+(size2/2)*t,y,temp+'0',size2);
	}
}

void OLED_ShowNum(uint8_t x,uint8_t y,unsigned int num,uint8_t len,uint8_t size2)
{
	uint8_t t,temp;
	uint8_t enshow=0;
	for(t=0;t<len;t++)
	{
		temp=(num/oled_pow(10,len-t-1))%10;
		if(enshow==0&&t<(len-1))
		{
			if(temp==0)
			{
				OLED_ShowChar(x+(size2/2)*t,y,' ',size2);
				continue;
			}else enshow=1;

		}
	 	OLED_ShowChar(x+(size2/2)*t,y,temp+'0',size2);
	}
}

void OLED_ShowChar(uint8_t x,uint8_t y,uint8_t chr,uint8_t Char_Size)
{
	unsigned char c=0,i=0;
		c=chr-' ';
		if(x>128-1){x=0;y=y+2;}
		if(Char_Size ==16)
			{
			OLED_Set_Pos(x,y);
			for(i=0;i<8;i++)
			OLED_WR_DATA(F8X16[c*16+i]);
			OLED_Set_Pos(x,y+1);
			for(i=0;i<8;i++)
			OLED_WR_DATA(F8X16[c*16+i+8]);
			}
			else {
				OLED_Set_Pos(x,y);
				for(i=0;i<6;i++)
				OLED_WR_DATA(F6x8[c][i]);

			}
}

void OLED_ShowString(uint8_t x,uint8_t y,uint8_t *chr,uint8_t Char_Size)
{
	unsigned char j=0;
	while (chr[j]!='\0')
	{		OLED_ShowChar(x,y,chr[j],Char_Size);
			x+=8;
		if(x>120){x=0;y+=2;}
			j++;
	}
}

void OLED_ShowCHinese(uint8_t x,uint8_t y,uint8_t no)
{
	uint8_t t,adder=0;
	OLED_Set_Pos(x,y);
    for(t=0;t<16;t++)
		{
				OLED_WR_DATA(Hzk[2*no][t]);
				adder+=1;
     }
		OLED_Set_Pos(x,y+1);
    for(t=0;t<16;t++)
			{
				OLED_WR_DATA(Hzk[2*no+1][t]);
				adder+=1;
      }
}

void OLED_Draw12864BMP(uint8_t num)
{
    uint16_t j=0;
    uint8_t x,y;
    for(y=0; y<8; y++)
    {
        OLED_Set_Pos(0,y);
        for(x=0; x<128; x++)
        {
#if DISPLAY_MODE
            OLED_WR_DATA(BMP[num-1][y]);
#else
            OLED_WR_DATA(BMP[num-1][j++]);
#endif
        }
    }
}

/* 清空指定行（页0~7），y 为页地址 */
void OLED_ClearLine(uint8_t y)
{
    uint8_t x;
    if (y > 7) return;
    OLED_Set_Pos(0, y);
    for (x = 0; x < 128; x++) {
        OLED_WR_DATA(0);
    }
}
